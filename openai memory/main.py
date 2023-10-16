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
uri = "<uri>"

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

unique_id = "<unique_id>(name)"

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"Tracing Walkthrough - {unique_id}"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "<langchain_api_key>" # Update to your API key #ls__

from dotenv import load_dotenv
load_dotenv(override=True)
