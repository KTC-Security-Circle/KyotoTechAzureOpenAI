# 必要なライブラリのインポート
from dotenv import load_dotenv
load_dotenv(override=True)
import os
os.environ["OPENAI_API_TYPE"]
os.environ["OPENAI_API_BASE"]
os.environ["OPENAI_API_KEY"]
os.environ["OPENAI_API_VERSION"]

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader

# 以下のコードで、テキストデータをvector store形式のDBを作成します。
embeddings = OpenAIEmbeddings(deployment=os.environ["DEPLOYMENT_EMBEDDINGS_NAME"], chunk_size=1) ## ここで、OpenAIのEmbeddingsを読み込みます。


directory_path = './dbDocs' # テキストデータのパス
persist_directory= 'splitDocsDB' # vector store形式のDBのパス

text_loader_kwargs={'autodetect_encoding': True} # テキストデータのエンコードを自動で判別する
loader = DirectoryLoader(path=directory_path, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs) # テキストデータの入ったフォルダを読み込みます。
docs = loader.load() # テキストデータを読み込みます。


vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_directory) # テキストデータをvector store形式のDBに変換します。
vectorstore.persist() # vector store形式のDBを保存します。
