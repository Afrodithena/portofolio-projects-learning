import os
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Loop semua PDF di folder
folder = "pdf_data"
all_docs = []
if not os.path.exists(folder):
    print(f"Error: Folder '{folder} tidak ditemukan!")
    exit()

print("--- Membaca Dokumen---")
for f in os.listdir(folder):
    if f.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(folder, f))
        all_docs.extend(loader.load())

# Split & Vectorize
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
raw_splits = splitter.split_documents(all_docs)

# Filter Sangat Ketat: 
# 1. Pastikan doc.page_content adalah STRING.
# 2. Pastikan tidak kosong setelah di-strip.
splits = []
for doc in raw_splits:
    if isinstance(doc.page_content, str) and doc.page_content.strip():
        # Pastikan tidak ada karakter aneh yang bikin encoder bingung
        clean_text = doc.page_content.encode("ascii", "ignore").decode("ascii")
        if len(clean_text.strip()) > 5: # Minimal 5 karakter agar tidak sampah
            doc.page_content = clean_text
            splits.append(doc)

print(f"--- Proses Embedding {len(splits)} potongan teks yang sudah divalidasi ---")

# --- BAGIAN VECTOR DB ---
embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Gunakan try-except supaya kalau satu gagal, kita tahu penyebabnya
try:
    vectorstore = Chroma.from_documents(documents=splits, embedding=embed_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
except Exception as e:
    print(f"Gagal saat proses Chroma: {e}")
    # Jika masih gagal, coba ambil 100 sample dulu untuk tes
    # vectorstore = Chroma.from_documents(documents=splits[:100], embedding=embed_model)

# RAG Chain
llm = OllamaLLM(model="llama3.1")
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
template = """Jawab pertanyaan berdasarkan konteks berikut:{context}
Pertanyaan: {question}
Jawaban:"""
prompt = ChatPromptTemplate.from_template(template)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
# Chat Loop
print("--- AI siap! (Ketik 'exit' untuk keluar) ---")
while True:
    q = input("User (ketik 'exit' untuk keluar): ")
    if q.lower() == 'exit': 
        break
    if q.strip() == "":
        continue
    print("AI sedang berpikir....")
    response = rag_chain.invoke(q)
    print(f"AI: {response}")