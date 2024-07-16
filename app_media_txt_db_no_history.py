import streamlit as st
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os


os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


def get_vectorstore_from_urls():
    vector_store = Chroma(persist_directory="./chroma_db_media_json", embedding_function=OpenAIEmbeddings())
    print(vector_store._collection.count())
    print(vector_store._collection)
    print('vector store loaded...')
    return vector_store


def get_context_retriever_chain(vector_store):

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={'score_threshold': 0.7, "k": 20}
    )


    prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user",
             "Generate a search query to the context to "
             "look up in order to get all the relevant information about user's question."
             " If the context is not enough, or unrelevant, write 'I do not know the answer.'")
        ])

    return retriever, prompt

def get_conversational_rag_chain(retriever, prompt):
    llm = ChatOpenAI()

    context_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions with short answers, based only on the below context:\n\n{context}. "
                   "If you can not find a relevant and proper answer in this context, tell the user that you do not know the answer."),
        ("user", "{input}"),
    ])



    stuff_documents_chain = create_stuff_documents_chain(llm, context_prompt)

    return create_retrieval_chain(retriever, stuff_documents_chain)

def get_response(user_input, vector_store):
    retriever, prompt = get_context_retriever_chain(vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever, prompt)
    print(st.session_state)
    response = conversation_rag_chain.invoke({
        "input": user_input
    })
    print('\n\n===> For the new response:')
    cnt = 1
    for resp in response['context']:
        print('-----------------')
        print(cnt, ' docs context: ', resp)
        cnt += 1

    if not response['context']:
        return "I do not know the answer."


    return response['answer']

# Streamlit app setup
st.set_page_config(page_title="Ask something about Mediastrom", page_icon="🦌")
st.title("Chat")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hello deer, I am a bot, I know everything about sleep and Mediastrom company. How can I help you?")]
if "vector_store" not in st.session_state:
    st.session_state.vector_store = get_vectorstore_from_urls()

user_query = st.text_input("Type your message here...")
submit_button = st.button("Submit")  # Add a submit button

if submit_button:  # Check if the submit button is pressed before processing the input
    if user_query:
        response = get_response(user_query, st.session_state.vector_store)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        st.write(f"🤖: {message.content}")
    elif isinstance(message, HumanMessage):
        st.write(f"**🦌: {message.content}**")