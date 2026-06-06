import streamlit as st
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# =========================
# LOAD ENV
# =========================

load_dotenv()

GROQ_API_KEY = "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"

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Intelligent Multi-Mode AI Agent",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Intelligent Multi-Mode AI Agent")

# =========================
# API KEY CHECK
# =========================

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found in .env file")
    st.stop()

# =========================
# LLM
# =========================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY
)

# =========================
# SESSION STATE
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# =========================
# VECTOR DB CREATION
# =========================

def create_vector_db(uploaded_file):

    text = uploaded_file.read().decode("utf-8")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = splitter.create_documents([text])

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = FAISS.from_documents(
        docs,
        embeddings
    )

    return vector_db

# =========================
# QUERY ROUTER
# =========================

def route_query(question):

    question = question.lower()

    rag_keywords = [
        "policy",
        "leave",
        "onboarding",
        "document",
        "proposal",
        "uploaded",
        "handbook",
        "hr",
        "company"
    ]

    for word in rag_keywords:
        if word in question:
            return "RAG"

    return "GENERAL"

# =========================
# SIDEBAR
# =========================

with st.sidebar:

    st.header("📂 Upload Document")

    uploaded_file = st.file_uploader(
        "Choose TXT File",
        type=["txt"]
    )

    if uploaded_file is not None:

        st.success(
            f"Uploaded: {uploaded_file.name}"
        )

        st.session_state.vector_db = create_vector_db(
            uploaded_file
        )

    st.markdown("---")

    st.header("⚙️ System Status")

    st.success("Memory Enabled")

    if st.session_state.vector_db:
        st.success("RAG Enabled")
    else:
        st.warning("No Document Uploaded")

    st.markdown("---")

    st.header("📜 Chat History")

    for msg in st.session_state.messages[-10:]:

        st.write(
            f"**{msg['role']}**: {msg['content'][:40]}"
        )

# =========================
# DISPLAY OLD CHATS
# =========================

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =========================
# USER INPUT
# =========================

question = st.chat_input(
    "Ask a question..."
)

if question:

    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    mode = route_query(question)

    with st.chat_message("assistant"):

        st.info(f"Mode: {mode}")

        # =====================
        # RAG MODE
        # =====================

        if (
            mode == "RAG"
            and st.session_state.vector_db is not None
        ):

            docs = st.session_state.vector_db.similarity_search(
                question,
                k=2
            )

            context = ""

            for doc in docs:
                context += doc.page_content + "\n"

            with st.expander(
                "Retrieved Context"
            ):
                st.write(context)

            prompt = f"""
You are an enterprise assistant.

Answer ONLY using the context below.

Context:
{context}

Question:
{question}
"""

            response = llm.invoke(prompt)

            answer = response.content

        # =====================
        # GENERAL MODE
        # =====================

        else:

            history = []

            for msg in st.session_state.messages:

                if msg["role"] == "user":

                    history.append(
                        HumanMessage(
                            content=msg["content"]
                        )
                    )

                else:

                    history.append(
                        AIMessage(
                            content=msg["content"]
                        )
                    )

            response = llm.invoke(history)

            answer = response.content

        st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )