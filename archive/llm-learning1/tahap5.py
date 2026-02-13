import streamlit as st
import os
import shutil
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 1. SETTING HALAMAN ---
st.set_page_config(page_title="Researcher AI Pro", page_icon="🕵️", layout="wide")
st.title("Researcher AI: Deep PDF Intelligence")

# --- 2. SESSION STATE (INGATAN) ---
if "messages" not in st.session_state:
    st.session_state.messages = [] 
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] 

with st.sidebar:
    st.header("Kontrol AI")
    if st.button("🗑️ Hapus Riwayat Chat"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()
    
    if st.button("Reset Database (Fix Error)"):
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")
        st.cache_resource.clear()
        st.success("Database dihapus! Silakan refresh halaman.")
        st.rerun()

# --- 3. FUNGSI KERJA (DENGAN FILTER KETAT) ---
@st.cache_resource
def build_knowledge_base(folder_path):
    persist_dir = "./chroma_db"
    embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if os.path.exists(persist_dir) and len(os.listdir(persist_dir)) > 0:
        return Chroma(persist_directory=persist_dir, embedding_function=embed_model).as_retriever(search_kwargs={"k": 4})

    if not os.path.exists(folder_path): return None
    
    all_docs = []
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    if not pdf_files: return None

    progress_bar = st.progress(0, text="Membaca file PDF...")
    for i, file in enumerate(pdf_files):
        loader = PyPDFLoader(os.path.join(folder_path, file))
        all_docs.extend(loader.load())
        progress_bar.progress((i + 1) / len(pdf_files))
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    raw_splits = splitter.split_documents(all_docs)
    
    splits = []
    for doc in raw_splits:
        content = str(doc.page_content) if doc.page_content else ""
        clean_content = content.encode("ascii", "ignore").decode("ascii").strip()
        if len(clean_content) > 10:
            doc.page_content = clean_content
            splits.append(doc)

    if not splits: return None

    try:
        vectorstore = Chroma.from_documents(documents=splits, embedding=embed_model, persist_directory=persist_dir)
        return vectorstore.as_retriever(search_kwargs={"k": 4})
    except Exception as e:
        st.error(f"Gagal membangun database: {e}")
        return None

# --- 4. PROSES AWAL ---
retriever = build_knowledge_base("pdf_data")

if retriever:
    llm = OllamaLLM(model="llama3.1")
    
    rewrite_template = """Perbaiki typo atau kesalahan ketik pada pertanyaan berikut agar menjadi kalimat
    tanya yang baku dan mudah dipahami, tanpa mengubah maksud aslinya.
    Langsung berikan hasil perbaikannya saja tanpa ada basa-basi.
    Pertanyaan: {input}"""
    rewrite_prompt = ChatPromptTemplate.from_template(rewrite_template)
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Anda analis riset senior. Gunakan konteks berikut untuk menjawab pertanyaan.\n\nKonteks: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # --- FIX: Menggunakan Lambda untuk memastikan retriever menerima String, bukan Dict ---
    rag_chain = (
        {
            # Tahap A: pertanyaan user (yang mungkin typo) akan dikoreksi dulu oleh rewriter_chain
            "input": rewrite_chain,
            "chat_history": lambda x: x["chat_history"]
        }
        | RunnablePassthrough.assign(
            # Tahap B: Pertanyaan yang sudah bersih dipakai buat nyari di database (Retriever)
            context= (lambda x: x["input"]) | retriever | format_docs
        )
        | prompt 
        | llm 
        | StrOutputParser()
    )

    # --- 5. TAMPILAN CHAT ---
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if query := st.chat_input("Tanya tentang ISLR, PDS, atau Deep Learning..."):
        # Tampilkan Pesan User
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            with st.spinner("Menganalisis dokumen..."):
                # Menjalankan stream dengan format input yang benar
                inputs = {"input": query, "chat_history": st.session_state.chat_history}
                for chunk in rag_chain.stream(inputs):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
            
            with st.expander("Lihat Sumber Referensi"):
                relevant_docs = retriever.invoke(query)
                for i, doc in enumerate(relevant_docs):
                    src = os.path.basename(doc.metadata.get('source', 'Unknown'))
                    page = doc.metadata.get('page', '?')
                    st.write(f"**Sumber {i+1}:** {src} (Halaman {page})")
        
        # Simpan History ke Session State
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.session_state.chat_history.append(("human", query))
        st.session_state.chat_history.append(("ai", full_response))
else:
    st.warning("Masukkan file PDF ke folder 'pdf_data' dulu!")


# Here is the evaluation result in JSON format:
#{
#  "conceptual_correctness": 4,
#  "coverage": 5,
#  "clarity": 4,
#  "depth": 3,
#  "overall_comment": "The answer provides a good overview of the essential concepts for understanding deep learning. The points made are generally correct, and the coverage is comprehensive. However, some of the explanations could be more detailed and nuanced. Overall, a solid effort."
#}
# 200s performance speednya naik