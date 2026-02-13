import os
from langchain_ollama import OllamaLLM # LLM yang dijalankan secara lokal lewat Ollama (contoh: Llama 3)
from langchain_huggingface import HuggingFaceEmbeddings # Model embedding dari HuggingFace [Embedding = mengubah teks → angka (vektor)]
from langchain_chroma import Chroma # Vector Database (penyimpanan embedding) [Dipakai supaya teks bisa dicari berdasarkan makna (semantic search)]
from langchain_community.document_loaders import PyPDFLoader # Loader untuk membaca file PDF dan mengubahnya jadi teks
from langchain_text_splitters import RecursiveCharacterTextSplitter # Memecah teks panjang jadi potongan kecil (chunk) [Supaya LLM tidak overload dan retrieval lebih akurat]
from langchain_core.prompts import ChatPromptTemplate # Template prompt (kerangka pertanyaan ke LLM) [Biar prompt konsisten & rapi]

from langchain_core.runnables import RunnablePassthrough 
from langchain_core.output_parsers import StrOutputParser

# --- 1. SETTING PATH ---
# Menggunakan r"" (raw string) agar backslash Windows tidak error
pdf_path = r"D:\portfolio-projects-learning\llm-learning\pdf_data\ISLR.pdf"

if not os.path.exists(pdf_path):
    print(f"Error: File tidak ditemukan di {pdf_path}")
    exit()

# --- 2. LOAD & SPLIT ---
print(f"Sedang membaca file: {os.path.basename(pdf_path)}...")
loader = PyPDFLoader(pdf_path)
docs = loader.load()

print(f"Memotong dokumen menjadi bagian-bagian kecil...")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
raw_splits = splitter.split_documents(docs)

# Filter: Bersihkan teks kosong/aneh agar Embedding tidak TypeError
splits = [doc for doc in raw_splits if isinstance(doc.page_content, str) and doc.page_content.strip()]
print(f"Berhasil membuat {len(splits)} potongan teks.")

# --- 3. VECTOR DB ---
print("Sedang proses Embedding (Teks -> Angka). Tunggu sebentar...")
# Gunakan persist_directory agar data tersimpan di folder lokal (opsional)
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
    persist_directory="./chroma_db_islr" 
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("Vector Database siap!")

# --- 4. RAG CHAIN ---
llm = OllamaLLM(model="llama3.1")

system_prompt = """Gunakan konteks berikut untuk menjawab pertanyaan pengguna. 
Tulis jawaban dalam tepat satu paragraf yang terdiri dari empat kalimat. 
Gunakan bahasa yang jelas, mengalir, dan cukup detail, namun tetap relevan dengan konteks. 
Jika jawabannya tidak ada di konteks, katakan dengan jujur bahwa kamu tidak tahu.

Konteks:
{context}

Pertanyaan:
{question}"""

prompt = ChatPromptTemplate.from_template(system_prompt)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG Chain (Pipe '|')
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- 5. TES & LOOP ---
print("\n" + "="*30)
print("AI Researcher SIAP!")
print("="*30)

while True:
    question = input("\nUser (ketik 'exit' untuk keluar): ")
    
    if question.lower() in ['exit', 'keluar', 'x']:
        print("Sampai jumpa!")
        break
    
    if not question.strip():
        continue

    print("Menghitung jawaban...", end="\r")
    answer = rag_chain.invoke(question)
    print("AI:" + " "*20) # Membersihkan teks "Menghitung..."
    print(answer)
