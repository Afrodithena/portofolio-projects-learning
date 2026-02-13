from langchain_ollama import OllamaLLM  # LLM lokal lewat Ollama (Llama 3)
from langchain_huggingface import HuggingFaceEmbeddings  # Mengubah teks jadi embedding/vektor
from langchain_community.vectorstores import Chroma # Vector Database (penyimpanan embedding) [Dipakai supaya teks bisa dicari berdasarkan makna (semantic search)]
from langchain_core.documents import Document # Mengimpor class Document, yaitu format standar LangChain [untuk merepresentasikan satu potongan data (teks + metadata)]

# 1. SIAPKAN DATA (Buku Contekan)
# Kita masukkan definisi RAG yang benar supaya Llama gak ngaco lagi
docs = [
    Document(page_content="RAG (Retrieval-Augmented Generation) adalah teknik dalam AI untuk memberikan konteks tambahan dari dokumen eksternal kepada LLM sebelum menghasilkan jawaban."),
    Document(page_content="Vector Database seperti ChromaDB atau Pinecone digunakan dalam RAG untuk menyimpan data dalam bentuk angka koordinat (vektor)."),
    Document(page_content="Proses RAG terdiri dari tiga tahap utama: Retrieval (mengambil data), Augmentation (menambah konteks), dan Generation (membuat jawaban).")
]

# 2. PROSES "SNAP" (Embedding)
# Kita pakai model gratis dari HuggingFace buat ngubah teks jadi angka
print("--- Sedang proses Embedding (Ubah teks jadi angka) ---")
embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 3. SIMPAN KE GUDANG (Vector DB)
# Kita simpan di folder lokal bernama 'my_db'
vector_db = Chroma.from_documents(
    documents=docs, 
    embedding=embed_model, 
    persist_directory="./my_db"
)
print("--- Data sudah tersimpan di Vector DB ---")

# 4. PROSES NYONTEK (Retrieval)
query = "Apa kepanjangan RAG dan apa fungsinya di AI?"
# Cari data yang paling relevan di gudang
contekkan = vector_db.similarity_search(query, k=1)
hasil_contekkan = contekkan[0].page_content

# 5. GENERATION (Jawab pakai contekan)
llm = OllamaLLM(model="llama3.1")

# Kita rakit perintahnya (Prompt)
prompt = f"""
Kamu adalah asisten AI yang jujur. 
Gunakan CONTEKKAN di bawah ini untuk menjawab PERTANYAAN user.
Jika tidak ada di contekan, katakan kamu tidak tahu.

CONTEKKAN: {hasil_contekkan}
PERTANYAAN: {query}
"""

print("\n--- Jawaban Llama (Setelah Nyontek RAG): ---")
print(llm.invoke(prompt))
