import streamlit as st
import os
import shutil
import time
import json

from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ================= PAGE =================
st.set_page_config(
    page_title="Researcher AI",
    page_icon="🧠",
    layout="wide"
)
st.title("Researcher AI")


# ================= SESSION =================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ================= SIDEBAR =================
with st.sidebar:
    st.header("Control System")

    if st.button("Reset Chat"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    if st.button("Reset Database"):
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")
        st.cache_resource.clear()
        st.success("Database dihapus.")
        st.rerun()


# ================= RETRIEVER =================
@st.cache_resource
def build_retriever(folder_path: str):
    persist_dir = "./chroma_db"

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        vectordb = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
        return vectordb.as_retriever(search_kwargs={"k": 3})

    docs = []
    if not os.path.exists(folder_path):
        return None

    for f in os.listdir(folder_path):
        if f.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, f))
            docs.extend(loader.load())

    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    splits = splitter.split_documents(docs)

    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    return vectordb.as_retriever(search_kwargs={"k": 3})


retriever = build_retriever("pdf_data")

if not retriever:
    st.warning("Masukkan PDF ke folder `pdf_data`.")
    st.stop()


# ================= LLM =================
llm = OllamaLLM(
    model="llama3.2:3b",
    temperature=0.15
)


# ================= PROMPT BUILDER =================
def build_prompt(context: str, history, question: str) -> str:
    history_text = ""
    for h in history[-4:]:
        role = "User" if h[0] == "human" else "Assistant"
        history_text += f"{role}: {h[1]}\n"

    return f"""
Anda adalah analis riset AI.

Aturan WAJIB:
- Gunakan HANYA konteks
- Fokus konsep inti + implikasi
- Jangan basa-basi
- Jangan generalisasi kosong

Konteks:
{context}

Riwayat singkat:
{history_text}

Pertanyaan:
{question}

Jawaban:
"""


# ================= GROUNDING CHECK =================
def grounded_check(answer: str, context: str) -> bool:
    ctx = set(context.lower().split())
    ans = set(answer.lower().split())
    return len(ctx & ans) >= 8


# ================= UI RENDER =================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ================= USER INPUT =================
if user_input := st.chat_input("Tanya tentang Deep Learning, ISLR, dll..."):
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    start = time.time()

    # 🔹 RETRIEVE ONCE
    docs = retriever.invoke(user_input)
    context_text = "\n\n".join(d.page_content for d in docs)

    prompt = build_prompt(
        context=context_text,
        history=st.session_state.chat_history,
        question=user_input
    )

    with st.chat_message("assistant"):
        with st.spinner("Menganalisis dokumen..."):
            answer = llm.invoke(prompt)

        elapsed = time.time() - start
        grounded = grounded_check(answer, context_text)

        st.markdown(answer)
        st.caption(
            f"{elapsed:.2f}s · "
            f"{'Grounded by documents' if grounded else 'Low confidence'}"
        )

        with st.expander("Referensi"):
            for i, d in enumerate(docs):
                src = os.path.basename(d.metadata.get("source", "Unknown"))
                page = d.metadata.get("page", "?")
                st.write(f"{i+1}. {src} (Hal. {page})")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    st.session_state.chat_history.extend([
        ("human", user_input),
        ("ai", answer)
    ])

# Here is the evaluation result in JSON format:
#{
#  "conceptual_correctness": 4,
#  "coverage": 3,
#  "clarity": 4,
#  "depth": 2,
#  "overall_comment": "The answer provides a clear and concise explanation of the concept, but could delve deeper into the implications of machine learning for non-experts. The discussion on the importance of statistical knowledge is well-represented, but some supporting evidence or examples would strengthen the argument."
#}
# 80s performance speednya naik