"""
=============================================================
  MedicQA - Processador de Livros (Rodar UMA VEZ no seu PC)
=============================================================
Este script:
1. Lê os PDFs dos livros (Guyton, Silverthorn, Porto)
2. Divide em chunks de texto
3. Gera embeddings vetoriais
4. Salva tudo em um arquivo .json para usar no servidor

Uso:
  1. Coloque os PDFs na pasta "livros/"
  2. Rode: python process_books.py
  3. Será gerado o arquivo "data/knowledge_base.json"
"""

import os
import json
import hashlib
import re
import sys
from pathlib import Path

# =============================================
# CONFIGURAÇÕES
# =============================================
BOOKS_DIR = "livros"          # Pasta onde estão os PDFs
OUTPUT_FILE = "data/knowledge_base.json"
CHUNK_SIZE = 800              # Caracteres por chunk
CHUNK_OVERLAP = 150           # Sobreposição entre chunks
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Modelo de embeddings (leve e eficiente)

# Mapeamento de nomes amigáveis dos livros
BOOK_NAMES = {
    "guyton": "Guyton - Tratado de Fisiologia Médica",
    "silverthorn": "Silverthorn - Fisiologia Humana",
    "porto": "Porto - Semiologia Médica",
}


def detect_book_name(filename: str) -> str:
    """Detecta qual livro é baseado no nome do arquivo."""
    fname = filename.lower()
    for key, name in BOOK_NAMES.items():
        if key in fname:
            return name
    return filename.replace(".pdf", "").replace("_", " ").title()


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extrai texto de um PDF usando PyPDF2."""
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        print("❌ PyPDF2 não encontrado. Instale com: pip install PyPDF2")
        sys.exit(1)

    print(f"  📖 Lendo: {pdf_path}")
    reader = PdfReader(pdf_path)
    text = ""
    total = len(reader.pages)
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        # Adiciona marcador de página para referência
        text += f"\n[PÁGINA {i + 1}]\n{page_text}"
        if (i + 1) % 50 == 0:
            print(f"    ... {i + 1}/{total} páginas processadas")
    print(f"    ✅ {total} páginas extraídas")
    return text


def clean_text(text: str) -> str:
    """Limpa o texto extraído do PDF."""
    # Remove múltiplos espaços e linhas em branco
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    # Remove caracteres estranhos de OCR
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\/\@\#\%\&\*\+\=\n\áéíóúãõâêîôûàèìòùçÁÉÍÓÚÃÕÂÊÎÔÛÀÈÌÒÙÇ°ºª]', '', text)
    return text.strip()


def chunk_text(text: str, book_name: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    """Divide o texto em chunks com sobreposição."""
    chunks = []
    current_page = 1

    # Divide por parágrafos primeiro
    paragraphs = text.split('\n\n')
    current_chunk = ""

    for para in paragraphs:
        # Detecta número da página
        page_match = re.search(r'\[PÁGINA (\d+)\]', para)
        if page_match:
            current_page = int(page_match.group(1))
            para = re.sub(r'\[PÁGINA \d+\]', '', para).strip()

        if not para.strip():
            continue

        if len(current_chunk) + len(para) <= chunk_size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk.strip():
                chunks.append({
                    "text": current_chunk.strip(),
                    "book": book_name,
                    "page": current_page,
                    "id": hashlib.md5(current_chunk.strip()[:100].encode()).hexdigest()[:12]
                })
            # Começa novo chunk com sobreposição
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else ""
            current_chunk = overlap_text + para + "\n\n"

    # Último chunk
    if current_chunk.strip():
        chunks.append({
            "text": current_chunk.strip(),
            "book": book_name,
            "page": current_page,
            "id": hashlib.md5(current_chunk.strip()[:100].encode()).hexdigest()[:12]
        })

    return chunks


def generate_embeddings(chunks: list) -> list:
    """Gera embeddings para todos os chunks usando sentence-transformers."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("❌ sentence-transformers não encontrado.")
        print("   Instale com: pip install sentence-transformers")
        sys.exit(1)

    print(f"\n🧠 Carregando modelo de embeddings: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c["text"] for c in chunks]
    print(f"🔢 Gerando embeddings para {len(texts)} chunks...")

    # Processa em batches para não estourar memória
    batch_size = 64
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings = model.encode(batch, show_progress_bar=False)
        all_embeddings.extend(embeddings.tolist())
        print(f"    ... {min(i + batch_size, len(texts))}/{len(texts)} processados")

    # Adiciona embeddings aos chunks
    for chunk, embedding in zip(chunks, all_embeddings):
        chunk["embedding"] = embedding

    return chunks


def main():
    print("=" * 60)
    print("  📚 MedicQA - Processador de Livros Médicos")
    print("=" * 60)

    # Verifica pasta de livros
    if not os.path.exists(BOOKS_DIR):
        os.makedirs(BOOKS_DIR)
        print(f"\n📁 Pasta '{BOOKS_DIR}/' criada!")
        print(f"   Coloque seus PDFs nela e rode novamente.")
        print(f"\n   Exemplo de nomes:")
        print(f"   - livros/guyton.pdf")
        print(f"   - livros/silverthorn.pdf")
        print(f"   - livros/porto.pdf")
        return

    # Lista PDFs
    pdfs = [f for f in os.listdir(BOOKS_DIR) if f.lower().endswith('.pdf')]
    if not pdfs:
        print(f"\n❌ Nenhum PDF encontrado em '{BOOKS_DIR}/'")
        print(f"   Coloque seus PDFs lá e rode novamente.")
        return

    print(f"\n📚 {len(pdfs)} livro(s) encontrado(s):")
    for pdf in pdfs:
        size_mb = os.path.getsize(os.path.join(BOOKS_DIR, pdf)) / (1024 * 1024)
        print(f"   • {pdf} ({size_mb:.1f} MB)")

    # Processa cada PDF
    all_chunks = []
    for pdf in pdfs:
        pdf_path = os.path.join(BOOKS_DIR, pdf)
        book_name = detect_book_name(pdf)

        print(f"\n{'─' * 50}")
        print(f"📖 Processando: {book_name}")
        print(f"{'─' * 50}")

        # Extrai texto
        text = extract_text_from_pdf(pdf_path)
        text = clean_text(text)

        # Divide em chunks
        chunks = chunk_text(text, book_name)
        print(f"  📝 {len(chunks)} chunks criados")

        all_chunks.extend(chunks)

    # Gera embeddings
    all_chunks = generate_embeddings(all_chunks)

    # Salva
    os.makedirs("data", exist_ok=True)
    output = {
        "model": EMBEDDING_MODEL,
        "embedding_dim": len(all_chunks[0]["embedding"]) if all_chunks else 0,
        "total_chunks": len(all_chunks),
        "books": list(set(c["book"] for c in all_chunks)),
        "chunks": all_chunks
    }

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False)

    file_size = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"\n{'=' * 60}")
    print(f"  ✅ PROCESSAMENTO CONCLUÍDO!")
    print(f"{'=' * 60}")
    print(f"  📊 Total de chunks: {len(all_chunks)}")
    print(f"  📚 Livros processados: {', '.join(output['books'])}")
    print(f"  💾 Arquivo salvo: {OUTPUT_FILE} ({file_size:.1f} MB)")
    print(f"\n  Próximo passo: rode 'python app.py' para iniciar o servidor!")


if __name__ == "__main__":
    main()
