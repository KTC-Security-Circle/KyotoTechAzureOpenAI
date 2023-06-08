from dotenv import load_dotenv
load_dotenv(override=True)
import os

from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from langchain.chains.question_answering import load_qa_chain

from langchain.memory import ConversationBufferMemory


embeddings = OpenAIEmbeddings(deployment=os.environ["DEPLOYMENT_EMBEDDINGS_NAME"])

vectordb = Chroma(persist_directory="splitDocsDB", embedding_function=embeddings)


llm = AzureChatOpenAI(
    openai_api_base=os.environ["OPENAI_API_BASE"],
    openai_api_version=os.environ["OPENAI_API_VERSION"],
    deployment_name=os.environ["DEPLOYMENT_NAME"],
    openai_api_key=os.environ["OPENAI_API_KEY"],
    openai_api_type = "azure",
)

# あなたは"京都デザイン＆テクノロジー専門学校"の教員です。学校に興味のあるユーザーからの質問に答えます。質問は音声をテキストに変換しているので、誤字や意味がおかしいものは修正してください。親しみを持ってもらえるように優しい言葉を使ってください。回答は日本語で回答してください。以下の情報を参考にして、質問に答えてください。回答は30文字から50文字程度で返答してください。もし与えられた情報量が多い場合は簡潔にまとめて、詳しく知りたい場合はもう一度質問させるように誘導してください。
template = """
system_message:
You are a teacher at "京都デザイン＆テクノロジー専門学校". You answer questions from users who are interested in the school. The questions are converted from audio to text, so please correct any typos or incorrect meanings. Please use friendly language to make it sound familiar. Please answer in Japanese. Please use the following information to answer the questions. Please answer in 60 to 80 characters. If the amount of information given is too much, please keep it brief and lead them to ask the question again if they want to know more details.

infomation:
{context}

chat_history:
{chat_history}

human_message:
{human_input}

AI_message:
"""
prompt = PromptTemplate(
    input_variables=["chat_history", "human_input", "context"], 
    template=template,
)
memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")
chain = load_qa_chain(
    llm=llm, 
    chain_type="stuff", 
    memory=memory, 
    prompt=prompt, 
    verbose=True
)

def ChatGptKyotoTechDocs(query):
    query = query
    docs = vectordb.similarity_search(query, k=1)
    result = chain({"input_documents": docs, "human_input": query})
    return result


'''
ここからTkinterでGUI画面の要素を定義していきます。
まず、GUI画面の土台を定義します。この後、画面上の各パーツを定義してセットすることを繰り返します。
'''
# GUI用パッケージtkinterを読み込みます。
from tkinter import *
root = Tk()

# 画面のサイズを定義
root.geometry("600x400")

# 画面のタイトルを定義
root.title("チャットボット試作版")

# フレーム(画面上の各パーツを配置するための枠)を定義
frame = Frame(root)


# スクロールバーを定義
sc = Scrollbar(frame)
# スクロールバーをセット
sc.pack(side=RIGHT, fill=Y)


# リストボックスを定義
msgs = Listbox(frame, width=80, height=20, yscrollcommand=sc.set)
# リストボックスをセット
msgs.pack(side=LEFT, fill=BOTH, pady=10)

#フレームをセット
frame.pack()


# テキストボックスを定義
textF = Entry(root, font=("Courier", 10),width=50)
# テキストボックスをセット
textF.pack()


# ボタンを押したときの動きについて関数を定義
def ask_from_bot():
    query = textF.get()
    msgs.insert(END, "you : " + query)
    msgs.update()
    result = ChatGptKyotoTechDocs(query)
    # chat_history = [(query, result["answer"])]
    # answer = result["answer"]
    msgs.insert(END, "bot : " + str(result["output_text"]))
    msgs.update()
    
    
    textF.delete(0, END)
    # msgs.yview(END)

# ボタンを定義
btn = Button(root, text="質問をどうぞ", font=("Courier", 10),bg='white', command=ask_from_bot)
# ボタンをセット
btn.pack()

# この関数を呼ぶとボタンを押す動きになる。
def enter_function(event):
    btn.invoke()

# エンターキーを押すとenter_functionを呼ぶ（つまり、エンターキーを押すとボタンを押すのと同じ動きになる。）
root.bind('<Return>', enter_function)


# TkinterでGUI画面を起動する。
root.mainloop()