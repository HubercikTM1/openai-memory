import pinecone
pinecone.init(api_key="<pinecone_api_key>", environment="gcp-starter")

import os
openai_api_key = "<openai_api_key>"

from langchain.embeddings import OpenAIEmbeddings
embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key)

from langchain.vectorstores import Pinecone


query = "Co się stanie z pieniędzmi które mam na koncie jak zamknę konto?"

index = Pinecone.from_existing_index("<pinecone-idx>", embeddings_model)


from pymongo.mongo_client import MongoClient

# how to get uri -> connect -> drivers -> python -> view full code example
uri = "<mongodb-uri>"

# Create a new client and connect to the server
client = MongoClient(uri)

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.callbacks import StdOutCallbackHandler
from langchain.memory import MongoDBChatMessageHistory
from langchain.memory import ConversationBufferMemory


message_history = MongoDBChatMessageHistory(
 connection_string=uri, session_id="test-session"
)
memory = ConversationBufferMemory(
 memory_key="chat_history", chat_memory=message_history
)

qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key),
 chain_type="stuff",
 retriever=index.as_retriever(),
 memory=memory, # add this line
 callbacks=[StdOutCallbackHandler()])

result = qa.run(query)

print(result)

unique_id = "<unique-id>"        #teams

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"Tracing Walkthrough - {unique_id}"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "<langchain_api_key>" # Update to your API key #ls__

from dotenv import load_dotenv
load_dotenv(override=True)


import os
import json

from langchain.utilities import SerpAPIWrapper
from langchain.tools import Tool
os.environ["SERPAPI_API_KEY"] = "<SERPAPI_API_KEY>"
search = SerpAPIWrapper()
search_tool = Tool(
    name="Google search",
    func=search.run,
    description="useful for when you need to ask with search"
 )
# test
answer = search_tool.run("kiedy są wybory parlamentarne")
print(json.dumps(answer, indent=4))


import requests
from langchain.tools import tool
@tool
def get_joke(query):
    """useful for when you need to tell a joke"""
    joke_url = f"https://v2.jokeapi.dev/joke/Any?type=single&contains={query}"
    response = requests.get(joke_url)
    if response.status_code == 200:
        joke_data = response.json()
        return joke_data['joke']
    else:
        raise Exception(f"Failed to retrieve joke: {response.status_code}")
print(get_joke("cat"))


from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
# ja to wrzuciłem do osobnego pliku, zauważ, że używam dekoratora na funkcji
from tools import get_joke, search_tool
my_openai_api_key = "<openai-api-key>"
chat_model = ChatOpenAI(openai_api_key=my_openai_api_key, temperature=0,
model='gpt-4')
memory = ConversationBufferMemory(memory_key="chat_history",
return_messages=True)
tools = [get_joke, search_tool]
agent = initialize_agent(tools, chat_model,
agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, memory=memory)

#test
query = "Tell me a joke about Java"
print(f"Got answer: {agent.run(query)}")
query = "kiedy są wybory parlamentarne"
print(f"Got answer: {agent.run(query)}")
