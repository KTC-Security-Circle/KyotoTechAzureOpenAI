from dotenv import load_dotenv
load_dotenv(override=True)
import os
os.environ["OPENAI_API_TYPE"]
os.environ["OPENAI_API_BASE"]
os.environ["OPENAI_API_KEY"]
os.environ["OPENAI_API_VERSION"]


import platform

import openai
import chromadb
import langchain

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader

embeddings = OpenAIEmbeddings(deployment=os.environ["DEPLOYMENT_EMBEDDINGS_NAME"], chunk_size=1)
# docsearch = Chroma.from_texts(texts, embeddings)


directory_path = './dbDocs'
persist_directory= 'splitDocsDB'

text_loader_kwargs={'autodetect_encoding': True}
loader = DirectoryLoader(path=directory_path, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
docs = loader.load()
# print(docs)

# for filename in file_list:
#     file_path = os.path.join(directory_path, filename)
#     if file_path.endswith(".txt"):
#         loader = TextLoader(file_path, encoding='utf8')
#         pages = loader.load_and_split()


vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_directory)
vectorstore.persist()
