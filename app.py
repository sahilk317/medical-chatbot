import os
from dotenv import load_dotenv
import streamlit as st
from store_index import Store_Index
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from src.prompt import system_prompt  
from langchain.schema.runnable import RunnableMap

# Load API key
load_dotenv()
# os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')

# Page config
st.set_page_config(page_title="MediBot - Medical Chatbot", page_icon="💊", layout="centered")

# Title
st.title("🩺 MediBot - Your Medical Assistant")

# Load index
store_index = Store_Index()
index_name = 'medical-chatbot'

# docsearch = store_index.Create_Index(index_name)

if index_name not in store_index.list_indexes():
    docsearch = store_index.Create_Index(index_name)
else:
    docsearch = store_index.Load_Index(index_name)

retriever = docsearch.as_retriever(search_type='similarity', search_kwargs={'k': 3})

# Load model
model = ChatGroq(model='llama3-70b-8192',api_key=groq_api_key)

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ('system', system_prompt),
    ('human', "Context:\n{context}\n\nQuestion: {question}")
])

# Initialize session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm MediBot. How can I help you today?"}
    ]

# Show previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input box
user_input = st.chat_input("Type your medical question here...")

def get_response_from_llm(query):
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])  # ✅ CHANGED: Extract page_content safely

    rag_chain = (
        RunnableMap({
            "question": lambda x: x,
            "context": lambda x: format_docs(retriever.invoke(x))  # ✅ CHANGED: format retriever output
        })
        | prompt
        | model
    )

    return rag_chain.invoke(query).content  # ✅ CHANGED: Extract raw string from AIMessage

mark = False
# If user types something
if user_input:
    mark = True
    st.session_state.messages.append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)

    # ✅ ADDED: Format documents before passing to prompt


    # Get response from model
    bot_response = get_response_from_llm(user_input)

    # ✅ FIXED: Always store message in consistent format
    st.session_state.messages.append({'role': 'assistant', 'content': bot_response})

    with st.chat_message('assistant'):
        st.markdown(bot_response)

