import atexit
import string
import uuid
import src.utils as utils
import logging
import subprocess
import sys
import signal
import sqlite3
import chromadb
from chromadb.utils import embedding_functions

class Memory:
    def __init__(self, config):
        self.config = config
        if self.config.vector_memory_enabled == '1':
            if self.config.vector_memory_chromadb_c_s == '1':
                # Start chromadb and don't wait 
                try:
                    self._db_process = subprocess.Popen(['chroma', 'run', '--path', self.config.vector_memory_db_path])
                except:
                    logging.error(f'Could not run chromadb. Mantella has no memory.')
                    input('\nPress any key to stop Mantella...')
                    sys.exit(0)
                # Try to terminate chromadb at program exit
                atexit.register(self.stop)
                signal.signal(signal.SIGINT, self.__signal_stop)
                signal.signal(signal.SIGTERM, self.__signal_stop)

                self._db_client = chromadb.HttpClient(host=self.config.vector_memory_db_host, port=self.config.vector_memory_db_port)
            else :
                self._db_client = chromadb.PersistentClient(path=self.config.vector_memory_db_path)

            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=self.config.vector_memory_embedding_model)
            self.sqlite_connection = sqlite3.connect(f'{self.config.vector_memory_db_path}/chroma.sqlite3')


    def __signal_stop(self, signum, frame):
        self.stop()

    def stop(self):
        '''Stop chromadb process if it's running'''
        if self.config.vector_memory_enabled == '1' and self._db_process is not None and self._db_process.poll() is None:
            self._db_process.terminate()

    @utils.time_it
    def memorize(self, convo_id, character_info, location, time, relationship='a stranger', character_comment='', player_comment='', summary='', type='snippet'):
        '''Memorize conversation snippet with provided metada'''
        if self.config.vector_memory_enabled == '0':
            return
        time_desc = utils.get_time_group(time)
        memory_str = f'It was {time_desc} in {location}.\n{character_info["name"]} was talking with {relationship} the player.\n{character_info["name"]} said: "{character_comment}".\nThe player responded: "{player_comment}"'
        if type == 'summary':
            memory_str = f'It was {time_desc} in {location} {character_info["name"]} was talking with {relationship} the player.\n {summary}'
        try:
            collection = self._db_client.get_or_create_collection(name=_collection_name(character_info['name']), metadata={"hnsw:space": "cosine"}, embedding_function=self.embedding_function)
            collection.add(documents=[memory_str], metadatas=[
                {'convo_id': convo_id, 'location': location, 'character': character_info['name'], 'type': type}
            ], ids=[uuid.uuid4().hex])
        except Exception as e:
            logging.error(f'Error saving memory to vectordb: {e}')

    @utils.time_it
    def recall(self, convo_id, character_info, location, time, relationship = 'a stranger', character_comment: str = None, player_comment: str = None):
        '''Recall memorized snippets. Provided metadata is used when constructing the query'''
        if self.config.vector_memory_enabled == '0':
            return None
        time_desc = utils.get_time_group(time)
        query_str =  f'{time_desc} in {location}.\n The player meets {relationship} {character_info["name"]}.'
        if player_comment is not None and len(player_comment) > 0:
            query_str = f'It is {time_desc} in {location}.\n{character_info["name"]} is talking with {relationship} the player.\n{character_info["name"]} said: "{character_comment}".\nThe player responds: {player_comment}"'
        try:
            collection = self._db_client.get_collection(name=_collection_name(character_info['name']), embedding_function=self.embedding_function)
            result = collection.query(query_texts=[query_str], 
                                      where={
                                          '$and': [
                                              {
                                                  'convo_id': {
                                                      '$ne': convo_id
                                                   },
                                              },
                                              {
                                                   'character': {
                                                       '$eq': character_info['name']
                                                   }
                                              }
                                          ]
                                      },
                                      include=['documents'])
            return result["documents"][0]
        except Exception as e:
            logging.error(f'Error loading memories from vectordb: {e}')
            return None
        
    def conversation_count(self, character_name):
        '''Return # of conversations character has had with player in the past'''
        if self.config.vector_memory_enabled == '0':
            return 0
        cur = self.sqlite_connection.cursor()
        res = cur.execute("select count(distinct b.string_value) as convo_id from embedding_metadata a join embedding_metadata b on (a.id=b.id and b.key='convo_id') where a.key='character' and a.string_value=:character", {"character": character_name})
        return res.fetchone()[0]

    def update_memories(self, message, memories):
        '''Append given memories to propmt that will be given to LLM'''
        if memories is not None and len(memories) > 0:
            mem = 'Below are your memories from past conversations:\n\n%s}' % "\n\n".join(memories)
            message = mem + '\n' + message
        return message

def _collection_name(character_name: str): 
    return character_name.lower().translate(str.maketrans('', '', string.punctuation + string.whitespace + string.digits))