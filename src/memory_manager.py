import asyncio
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
        self.enabled = self.config.vector_memory_enabled == '1'
        if self.enabled:
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
            # Initialize sqlite3 connection for querying by metadata
            self.sqlite_connection = sqlite3.connect(f'{self.config.vector_memory_db_path}/chroma.sqlite3')


    def __signal_stop(self, signum, frame):
        self.stop()

    def stop(self):
        '''Stop chromadb process if it's running'''
        if self.config.vector_memory_enabled == '1' and self._db_process is not None and self._db_process.poll() is None:
            self._db_process.terminate()

    @utils.time_it
    def memorize(self, convo_id, character, location, time, messages=[], summary='', type='fragment'):
        '''Memorize conversation fragment (fragment or symmary) with provided metada. 
           Fragments are summarized from 4 last comments in the conversation transcript'''
        if not self.enabled:
            return
        if len(messages[3:]) >= 4:
            loop = asyncio.get_running_loop()
            loop.create_task(self._summarize_and_store(convo_id, character, location, time, messages, summary, type))

    async def _summarize_and_store(self, convo_id, character, location, time, messages, summary='', type='fragment') :
        # Summarize 2 latest interactions starting after greetings
        if type == 'fragment' and len(messages[3:]) > 4:
            summary = character.summarize_conversation(messages[-4:], self.config.llm, 4, is_fragment=True)
        time_desc = utils.get_time_group(time)
        relationship = utils.get_trust_desc(self.conversation_count(character.info['name']), character.relationship_rank) 
        memory_str = f'{time_desc} in {location}. {summary}'
        collection = self._db_client.get_or_create_collection(name=_collection_name(character.info['name']), metadata={"hnsw:space": "cosine"}, embedding_function=self.embedding_function)
        collection.add(documents=[memory_str], metadatas=[
            {'convo_id': convo_id, 'location': location, 'character': character.info['name'], 'type': type}
        ], ids=[uuid.uuid4().hex])

    @utils.time_it
    def recall(self, convo_id, character, location, time, messages, player_comment: str = None):
        '''Update the prompt at the start of the conversation to contain new memories related to the most recent comments in the conversation'''
        if not self.enabled:
            return None
        time_desc = utils.get_time_group(time)
        relationship = utils.get_trust_desc(self.conversation_count(character.info['name']), character.relationship_rank) 
        query_str =  f'{time_desc} in {location}.\n The player meets {relationship} {character.info["name"]}.'
        if player_comment is not None and len(player_comment) > 0:
            query_str = f'It is {time_desc} in {location}.\n{character.info["name"]} is talking with {relationship} the player.\n{character.info["name"]} said: "{messages[-1]["content"]}".\nThe player responds: {player_comment}"'

        logging.info(f'Finding {character.info["name"]}\'s memories using query "{query_str}"')
        collection = self._db_client.get_or_create_collection(name=_collection_name(character.info['name']), embedding_function=self.embedding_function)
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
                                                    '$eq': character.info['name']
                                                }
                                            }
                                        ]
                                    },
                                    n_results=5,
                                    include=['documents'])
        memories = result["documents"][0]
        mem = "You have never met the player before"
        if len(memories) > 0 :
            mem = 'Below are your memories from past conversations:\n\n%s\n\n' % "\n\n".join(memories)
        context = character.create_context(self.config.prompt, location, time, {character.name: character}, self.config.custom_token_count, self.conversation_count(character.info['name']), mem, convo_id=convo_id)
        messages[0]['content'] = context[0]['content']
        
    def conversation_count(self, character_name):
        '''Return # of conversations character has had with player in the past'''
        if not self.enabled:
            return 0
        cur = self.sqlite_connection.cursor()
        query = """
select count(distinct b.string_value) as convo_id 
from 
    embedding_metadata a 
    join embedding_metadata b on (a.id=b.id and b.key='convo_id') 
where 
    a.key='character' 
    and a.string_value=:character
"""
        res = cur.execute(query, {"character": character_name})
        num = res.fetchone()[0]
        logging.info(f'{character_name} has {num} conversations with player')
        return num

def _collection_name(character_name: str): 
    return character_name.lower().translate(str.maketrans('', '', string.punctuation + string.whitespace + string.digits))