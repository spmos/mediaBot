import streamlit as st
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


def get_vectorstore_from_urls(vector_store_path=None):

    vector_store = Chroma(persist_directory=vector_store_path, embedding_function=OpenAIEmbeddings())
    print(vector_store._collection.count())
    #print(vector_store._collection)
    print('vector store loaded...')
    return vector_store

def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()
    #print(vector_store)
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={'score_threshold': 0.6, "k": 20}
    )
    # retriever = vector_store.as_retriever(
    #     search_type="mmr",
    #     search_kwargs={'k': 8, 'lambda_mult': 0.25, 'fetch_k': 20}
    # )
    prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user",
             "Given the above conversation, generate a search query to the context to "
             "look up in order to get all the relevant information about user's question.")

        ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain


def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions with short answers, based only on the below context:\n\n{context}. "
                   "Given your expertise in the company Mediastrom, your responses should be strictly limited to information about the company and the provided context."
                   " Please ensure your answers are concise and directly related to context, avoiding any deviations or unrelated content."
                   "If you can not find a relevant and proper answer in this context, tell the user that you do not know the answer."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def get_response(user_input, vector_store):
    retriever_chain = get_context_retriever_chain(vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    #print(st.session_state.chat_history)
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    print('\n\n===> For the new response:')
    cnt = 1
    for resp in response['context']:
        print('-----------------')
        print(cnt, ' docs context: ', resp)
        cnt += 1
    if not response['context']:
        return "I have not information about that, I am sorry! 🤔"

    return response['answer']

# Streamlit app setup
st.set_page_config(page_title="Ask something about Mediastrom", page_icon="🦌")
st.title("Chat")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hello deer, I am a bot, I know everything about sleep and Mediastrom company. How can I help you?")]
if "vector_store" not in st.session_state:
    st.session_state.vector_store = get_vectorstore_from_urls('./chroma_db_media_txt')

#user_query = st.text_input("Type your message here...")
#submit_button = st.button("Submit")  # Add a submit button

with st.form(key='user_query_form'):
    user_query = st.text_input("Type your message here...")
    submit_button = st.form_submit_button("Submit", type="primary")

col1, col2, col3, col4 = st.columns([3, 2, 2, 2])  # Create columns for layout
with col1:
    sleep_coach_button = st.button("Sleep Coach")
with col2:
    beds_button = st.button("Beds")
with col3:
    mattresses_button = st.button("Mattresses")
with col4:
    pillows_button = st.button("Pillows")

if sleep_coach_button:
    st.session_state.chat_history = [AIMessage(
        content="Hello deer, I am a bot, I know everything about sleep and Mediastrom company. How can I help you?")]
    st.session_state.vector_store = get_vectorstore_from_urls('./chroma_db_media_txt')
if beds_button:
    st.session_state.chat_history = [AIMessage(
        content="Hello deer, I am a bot, I know everything about Mediastrom company and its beds. How can I help you?")]
    st.session_state.vector_store = get_vectorstore_from_urls('./chroma_db_media_beds')
if mattresses_button:
    st.session_state.chat_history = [AIMessage(
        content="Hello deer, I am a bot, I know everything about Mediastrom company and its mattresses. How can I help you?")]
    st.session_state.vector_store = get_vectorstore_from_urls('./chroma_db_media_mattresses')
if pillows_button:
    st.session_state.chat_history = [AIMessage(
        content="Hello deer, I am a bot, I know everything about Mediastrom company and its pillows. How can I help you?")]
    st.session_state.vector_store = get_vectorstore_from_urls('./chroma_db_media_pillows')




if submit_button:  # Check if the submit button is pressed before processing the input
    if user_query:
        response = get_response(user_query, st.session_state.vector_store)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

# Function to handle thumbs up and thumbs down


for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        st.write(f"🤖: {message.content}")
    elif isinstance(message, HumanMessage):
        st.write(f"**🦌: {message.content}**")


# ---------------------- WITHOUT HISTORY AWARE RETRIEVER ---------------------- #


#
# import streamlit as st
# from langchain_core.messages import AIMessage, HumanMessage
# from langchain_community.document_loaders import WebBaseLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from dotenv import load_dotenv
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
#
# load_dotenv()
#
# def get_vectorstore_from_urls():
#     vector_store = Chroma(persist_directory="./chroma_db_media_txt", embedding_function=OpenAIEmbeddings())
#     print(vector_store._collection.count())
#     print(vector_store._collection)
#     print('vector store loaded...')
#     return vector_store
#
# def get_context_retriever_chain(vector_store):
#     llm = ChatOpenAI()
#     print(vector_store)
#     retriever = vector_store.as_retriever()
#
#     prompt = ChatPromptTemplate.from_messages([
#         ("user", "{input}"),
#         ("user", "Generate a search query to retrieve the context needed to answer the user's question.")
#     ])
#
#     return retriever, prompt
#
# def get_conversational_rag_chain(retriever, prompt):
#     llm = ChatOpenAI()
#
#     context_prompt = ChatPromptTemplate.from_messages([
#         ("system", "Answer the user's questions with details, based only on the below context:\n\n{context}. "
#                    "If you cannot find a relevant and proper answer in this context, tell the user that you do not know the answer."),
#         ("user", "{input}"),
#     ])
#
#     stuff_documents_chain = create_stuff_documents_chain(llm, context_prompt)
#
#     return create_retrieval_chain(retriever, stuff_documents_chain)
#
# def get_response(user_input, vector_store):
#     retriever, prompt = get_context_retriever_chain(vector_store)
#     conversation_rag_chain = get_conversational_rag_chain(retriever, prompt)
#     print(st.session_state)
#     response = conversation_rag_chain.invoke({
#         "input": user_input
#     })
#
#     return response['answer']
#
# # Streamlit app setup
# st.set_page_config(page_title="Ask something about Mediastrom", page_icon="🦌")
# st.title("Chat")
#
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = [AIMessage(content="Hello deer, I am a bot, I know everything about sleep and Mediastrom company. How can I help you?")]
# if "vector_store" not in st.session_state:
#     st.session_state.vector_store = get_vectorstore_from_urls()
#
# user_query = st.text_input("Type your message here...")
# submit_button = st.button("Submit")  # Add a submit button
#
# if submit_button:  # Check if the submit button is pressed before processing the input
#     if user_query:
#         response = get_response(user_query, st.session_state.vector_store)
#         st.session_state.chat_history.append(HumanMessage(content=user_query))
#         st.session_state.chat_history.append(AIMessage(content=response))
#
# for message in st.session_state.chat_history:
#     if isinstance(message, AIMessage):
#         st.write(f"🤖: {message.content}")
#     elif isinstance(message, HumanMessage):
#         st.write(f"**🦌: {message.content}**")
#
