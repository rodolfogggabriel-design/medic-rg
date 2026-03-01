"""
=============================================================
  MedicQA v2 - Servidor Web com Multi-API Rotation
=============================================================
Combina TODAS as APIs gratuitas (Gemini, Groq, Mistral, 
Cohere, HuggingFace, Together, OpenRouter) com rotação 
automática. Resultado: uso praticamente ilimitado, custo zero.

Uso:
  python app.py
  Acesse: http://localhost:5000
"""

import os
import json
import numpy as np
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import requests
import time

from multi_api import MultiAPIManager

load_dotenv()

app = Flask(__name__)

# =============================================
# CONFIGURAÇÕES
# =============================================
KNOWLEDGE_BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'knowledge_base.json')
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
TOP_K = 2
MAX_HISTORY = 10

# =============================================
# INICIALIZAÇÃO
# =============================================
knowledge_base = None
api_manager = MultiAPIManager()

SYSTEM_PROMPT = """Você é o MedicQA, um assistente de estudos médicos especializado.
Você responde perguntas com base nos livros de medicina fornecidos como contexto.

REGRAS:
1. Responda SEMPRE em português brasileiro
2. Base suas respostas EXCLUSIVAMENTE no contexto fornecido dos livros
3. Cite o livro e a página quando referenciar informações específicas (ex: "Segundo Guyton, p.45...")
4. Se o contexto não contiver informação suficiente, diga claramente
5. Use linguagem acessível mas tecnicamente precisa
6. Organize a resposta com clareza (use parágrafos, não listas excessivas)
7. Se relevante, faça conexões entre conceitos de diferentes livros
8. Ao final, sugira tópicos relacionados que o aluno pode estudar

CONTEXTO DOS LIVROS:
"""


def load_knowledge_base():
    global knowledge_base
    if not os.path.exists(KNOWLEDGE_BASE_PATH):
        print("⚠️  Base de conhecimento não encontrada!")
        print(f"   Rode primeiro: python process_books.py")
        knowledge_base = {"chunks": [], "books": [], "total_chunks": 0}
        return

    print("📚 Carregando base de conhecimento...")
    with open(KNOWLEDGE_BASE_PATH, 'r', encoding='utf-8') as f:
        knowledge_base = json.load(f)

    for chunk in knowledge_base["chunks"]:
        chunk["embedding"] = np.array(chunk["embedding"])

    print(f"   ✅ {knowledge_base['total_chunks']} chunks carregados")
    print(f"   📚 Livros: {', '.join(knowledge_base['books'])}")


def cosine_similarity(a, b):
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot / (norm_a * norm_b)


def search_similar_chunks(query_embedding, top_k=TOP_K):
    if not knowledge_base or not knowledge_base["chunks"]:
        return []

    similarities = []
    for chunk in knowledge_base["chunks"]:
        sim = cosine_similarity(query_embedding, chunk["embedding"])
        similarities.append((sim, chunk))

    similarities.sort(key=lambda x: x[0], reverse=True)

    results = []
    for sim, chunk in similarities[:top_k]:
        results.append({
            "text": chunk["text"],
            "book": chunk["book"],
            "page": chunk["page"],
            "similarity": float(sim)
        })
    return results


def get_query_embedding(query: str):
    if GEMINI_API_KEY and GEMINI_API_KEY != "cole_sua_chave_aqui":
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={GEMINI_API_KEY}"
            payload = {
                "model": "models/text-embedding-004",
                "content": {"parts": [{"text": query}]}
            }
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code == 200:
                return np.array(resp.json()["embedding"]["values"])
        except Exception as e:
            print(f"⚠️  Gemini embedding falhou: {e}")

    try:
        from sentence_transformers import SentenceTransformer
        if not hasattr(get_query_embedding, '_model'):
            model_name = knowledge_base.get("model", "all-MiniLM-L6-v2")
            get_query_embedding._model = SentenceTransformer(model_name)
        return np.array(get_query_embedding._model.encode(query))
    except ImportError:
        return None


# =============================================
# ROTAS
# =============================================
@app.route("/")
def index():
    books = knowledge_base.get("books", []) if knowledge_base else []
    total = knowledge_base.get("total_chunks", 0) if knowledge_base else 0
    return render_template("index.html", books=books, total_chunks=total)


@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "").strip()
    history = data.get("history", [])

    if not question:
        return jsonify({"error": "Pergunta vazia"}), 400

    start = time.time()

    # 1. Embedding
    query_emb = get_query_embedding(question)
    if query_emb is None:
        return jsonify({
            "answer": "❌ Não foi possível gerar embedding. Verifique a configuração.",
            "sources": [], "time": 0, "provider": "none"
        })

    # 2. Busca semântica
    results = search_similar_chunks(query_emb, top_k=TOP_K)

    # 3. Contexto
    context_parts = []
    for i, chunk in enumerate(results, 1):
        context_parts.append(f"[Trecho {i} - {chunk['book']}, p.{chunk['page']}]\n{chunk['text']}")
    context = "\n\n---\n\n".join(context_parts)[:4000]
    system_prompt = SYSTEM_PROMPT + context[:1500]

    # 4. Mensagens
    messages = []
    if history:
        for msg in history[-MAX_HISTORY:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": question})

    # 5. Multi-API com rotação!
    response = api_manager.generate(system_prompt, messages)

    elapsed = time.time() - start

    # 6. Fontes
    sources = []
    seen = set()
    for r in results:
        key = f"{r['book']}-p{r['page']}"
        if key not in seen:
            sources.append({
                "book": r["book"],
                "page": r["page"],
                "relevance": round(r["similarity"] * 100, 1)
            })
            seen.add(key)

    return jsonify({
        "answer": response["text"],
        "sources": sources,
        "time": round(elapsed, 2),
        "provider": response["provider"],
        "fallback": response["fallback"]
    })


@app.route("/api/status")
def status():
    api_status = api_manager.get_status()

    kb_status = "empty" if (not knowledge_base or not knowledge_base["chunks"]) else "ready"

    return jsonify({
        "status": kb_status,
        "books": knowledge_base.get("books", []) if knowledge_base else [],
        "total_chunks": knowledge_base.get("total_chunks", 0) if knowledge_base else 0,
        "apis": api_status
    })


@app.route("/api/providers")
def providers():
    return jsonify(api_manager.get_status())


# =============================================
# INIT
# =============================================
load_knowledge_base()
    print("=" * 60)
    print("  🏥 MedicQA v2 — Multi-API Free Tier Rotation")
    print("=" * 60)

    load_knowledge_base()

    print(f"\n🌐 Servidor: http://localhost:5000")
    print(f"   Ctrl+C para parar\n")

    app.run(host="0.0.0.0", port=5000, debug=True)
