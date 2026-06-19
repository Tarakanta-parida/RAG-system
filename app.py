import os
import tempfile
from pathlib import Path
from time import sleep

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

APP_TITLE = "Document QnA Bot"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CHAT_MODEL = os.getenv("GOOGLE_CHAT_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "gemini-embedding-2-preview")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "3"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))

st.set_page_config(page_title=APP_TITLE, layout="centered")

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False


def reset_document():
    st.session_state.vector_db = None
    st.session_state.messages = []
    st.session_state.document_uploaded = False


def require_google_api_key():
    if GOOGLE_API_KEY:
        return True

    st.error("GOOGLE_API_KEY is not configured. Add it to your deployment environment.")
    st.stop()


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGoogleGenerativeAI(model=CHAT_MODEL, google_api_key=GOOGLE_API_KEY)


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )


def build_prompt(context, query):
    return (
        "You are a helpful assistant. Answer using only the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}"
    )


def document_process(pdf_path):
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()

    if not docs:
        raise ValueError("The PDF did not contain readable text.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)

    if not chunks:
        raise ValueError("The PDF could not be split into searchable text chunks.")

    vector_db = InMemoryVectorStore.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
    )

    st.session_state.vector_db = vector_db
    st.session_state.document_uploaded = True


def save_upload_to_temp_file(uploaded_file):
    suffix = Path(uploaded_file.name).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return Path(tmp_file.name)


require_google_api_key()

st.title(APP_TITLE)
st.caption("Upload a PDF and ask questions grounded in the document.")

with st.sidebar:
    st.header("Document")
    if st.session_state.document_uploaded:
        st.success("PDF processed")
        st.button("Upload another PDF", on_click=reset_document, use_container_width=True)
    else:
        st.info("No PDF uploaded")

if not st.session_state.document_uploaded:
    file = st.file_uploader(
        "Select your PDF file",
        type=["pdf"],
        accept_multiple_files=False,
    )

    if file:
        upload_size_mb = len(file.getvalue()) / (1024 * 1024)
        if upload_size_mb > MAX_UPLOAD_MB:
            st.error(f"PDF is too large. Maximum upload size is {MAX_UPLOAD_MB} MB.")
            st.stop()

        pdf_path = save_upload_to_temp_file(file)
        try:
            with st.spinner("Processing the document..."):
                document_process(pdf_path)
        except Exception as exc:
            st.error(f"Could not process this PDF: {exc}")
            st.stop()
        finally:
            pdf_path.unlink(missing_ok=True)

        st.success("Document processed successfully. You can now ask questions.")
        sleep(1)
        st.rerun()

if st.session_state.document_uploaded and st.session_state.vector_db is not None:
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    query = st.chat_input("Ask anything related to the document:")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").markdown(query)

        with st.spinner("Searching the document..."):
            documents = st.session_state.vector_db.similarity_search(query, k=RETRIEVAL_K)
            context = "\n\n".join(doc.page_content for doc in documents)
            prompt = build_prompt(context, query)
            result = get_llm().invoke(prompt)

        answer = result.content
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").markdown(answer)
