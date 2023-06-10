# 以下のコードにコメントをいれていきます。
# まず、必要なパッケージを読み込みます。
from dotenv import load_dotenv
load_dotenv(override=True)
import os

from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from langchain.prompts import (
    PromptTemplate,
)

from langchain.chains.question_answering import load_qa_chain

from langchain.memory import ConversationBufferMemory

# 以下のコードで、AzureのOpenAIを使って、京都デザイン＆テクノロジー専門学校の質問に答えるチャットボットを作成します。
embeddings = OpenAIEmbeddings(deployment=os.environ["DEPLOYMENT_EMBEDDINGS_NAME"]) ## ここで、OpenAIのEmbeddingsを読み込みます。
vectordb = Chroma(persist_directory="splitDocsDB", embedding_function=embeddings)  ## ここで、Chromaを読み込みます。

llm = AzureChatOpenAI(
    openai_api_base=os.environ["OPENAI_API_BASE"],
    openai_api_version=os.environ["OPENAI_API_VERSION"],
    deployment_name=os.environ["DEPLOYMENT_NAME"],
    openai_api_key=os.environ["OPENAI_API_KEY"],
    openai_api_type = "azure",
) ## ここで、AzureのOpenAIを使ってllmを作成します。

## 以下のテンプレートを使って、チャットボットの回答を作成します。
### あなたは"京都デザイン＆テクノロジー専門学校"の教員です。学校に興味のあるユーザーからの質問に答えます。質問は音声をテキストに変換しているので、誤字や意味がおかしいものは修正してください。親しみを持ってもらえるように優しい言葉を使ってください。回答は日本語で回答してください。以下の情報を参考にして、質問に答えてください。回答は30文字から50文字程度で返答してください。もし与えられた情報量が多い場合は簡潔にまとめて、詳しく知りたい場合はもう一度質問させるように誘導してください。
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
## 以下のコードで、チャットボットの回答を作成します。
prompt = PromptTemplate(
    input_variables=["chat_history", "human_input", "context"], 
    template=template,
) ## ここで、PromptTemplateを作成します。
memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input") ## ここで、ConversationBufferMemoryを作成します。
chain = load_qa_chain(
    llm=llm, 
    chain_type="stuff", 
    memory=memory, 
    prompt=prompt, 
    verbose=True
) ## ここで、load_qa_chainを使って、チャットボットの回答を作成します。

def ChatGptKyotoTechDocs(query): ## ここで、ChatGptKyotoTechDocsを定義します。
    query = query
    docs = vectordb.similarity_search(query, k=3)
    result = chain({"input_documents": docs, "human_input": query})
    return result


# ここからTkinterでGUI画面の要素を定義していきます。
## GUI用パッケージtkinterを読み込みます。
from tkinter import *
root = Tk()

## 画面のサイズを定義
root.geometry("700x400")

## 画面のタイトルを定義
root.title("京都テック君試作版")

## フレーム(画面上の各パーツを配置するための枠)を定義
frame = Frame(root)


## スクロールバーを定義
sc = Scrollbar(frame)
## スクロールバーをセット
sc.pack(side=RIGHT, fill=Y)


# ## リストボックスを定義
# msgs = Listbox(frame, width=80, height=20, yscrollcommand=sc.set)
# ## リストボックスをセット
# msgs.pack(side=LEFT, fill=BOTH, pady=10)

## リストボックスを定義
msgs = Listbox(frame, width=100, height=20, yscrollcommand=sc.set, xscrollcommand=sc.set)
## リストボックスをセット
msgs.pack(side=LEFT, fill=BOTH, pady=10)

## フレームをセット
frame.pack()


## テキストボックスを定義
textF = Entry(root, font=("Courier", 10),width=50)
## テキストボックスをセット
textF.pack()


## ボタンを押したときの動きについて関数を定義
def ask_from_bot():
    query = textF.get() ## テキストボックスに入力された文字を取得
    textF.delete(0, END) ## テキストボックスの文字を削除
    msgs.insert(END, "you : " + query) ## テキストボックスに入力された文字をリストボックスに表示
    msgs.insert(END, "bot : 考え中...") ## リストボックスに「考え中...」と表示
    msgs.update() ## リストボックスを更新
    result = ChatGptKyotoTechDocs(query) ## チャットボットの回答を作成しresultに格納
    msgs.delete(END) ## リストボックスの最後の行を削除
    response = str(result["output_text"]) ## チャットボットのresult["output_text"]をresponseに格納
    if len(response) < 50: ## チャットボットの回答が50文字以下の場合
        msgs.insert(END, "bot : " + response) ## チャットボットの回答をリストボックスに表示
    else: ## チャットボットの回答が50文字より多い場合
        msgs.insert(END, "bot : " + response[:50]) ## チャットボットの回答の最初の50文字をリストボックスに表示
        response = response[50:] ## チャットボットの回答の最初の50文字を削除
        while len(response) > 50: ## まだチャットボットの回答が50文字より多い場合
            msgs.insert(END, "         " + response[:50]) ## チャットボットの回答の次の50文字をリストボックスに表示
            response = response[50:] ## チャットボットの回答の次の50文字を削除
        if response: ## チャットボットの回答が残っている場合
            msgs.insert(END,"          " + response) ## チャットボットの回答をリストボックスに表示
    msgs.update() ## リストボックスを更新

## ボタンを定義
btn = Button(root, text="質問をどうぞ", font=("Courier", 10),bg='white', command=ask_from_bot)
## ボタンをセット
btn.pack()

## この関数を呼ぶとボタンを押す動きになる。
def enter_function(event):
    btn.invoke()

## エンターキーを押すとenter_functionを呼ぶ（つまり、エンターキーを押すとボタンを押すのと同じ動きになる。）
root.bind('<Return>', enter_function)


## TkinterでGUI画面を起動する。
root.mainloop()