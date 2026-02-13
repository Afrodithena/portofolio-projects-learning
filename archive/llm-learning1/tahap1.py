from langchain_ollama import OllamaLLM  # LLM yang dijalankan secara lokal lewat Ollama (contoh: Llama 3)

# Inisialisasi model yang sudah diinstall di Ollama
model = OllamaLLM(model="llama3.1")

# Jalankan prompt
print("--- Menunggu Jawaban Llama ---")
response = model.invoke("Jelaskan singkat apa itu RAG?")
print(response)