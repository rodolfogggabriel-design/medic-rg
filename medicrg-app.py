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

KNOWLEDGE_BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'knowledge_base.json')
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
TOP_K = 2
MAX_HISTORY = 10

knowledge_base = None
api_manager = MultiAPIManager()

SYSTEM_PROMPT = "Voce e o MEDIC.RG, um assistente de estudos medicos especializado.\nVoce responde perguntas com base nos livros de medicina fornecidos como contexto.\n\nREGRAS:\n1. Responda SEMPRE em portugues brasileiro\n2. Base suas respostas EXCLUSIVAMENTE no contexto fornecido dos livros\n3. Cite o livro e a pagina quando referenciar informacoes especificas\n4. Se o contexto nao contiver informacao suficiente, diga claramente\n5. Use linguagem acessivel mas tecnicamente precisa\n6. Organize a resposta com clareza\n7. Se relevante, faca conexoes entre conceitos de diferentes livros\n8. Ao final, sugira topicos relacionados que o aluno pode estudar\n\nCONTEXTO DOS LIVROS:\n"


def load_knowledge_base():
    global knowledge_base
    if not os.path.exists(KNOWLEDGE_BASE_PATH):
        print("Base nao encontrada: " + KNOWLEDGE_BASE_PATH)
        knowledge_base = {"chunks": [], "books": [], "total_chunks": 0}
        return
    print("Carregando base de conhecimento...")
    with open(KNOWLEDGE_BASE_PATH, 'r', encoding='utf-8') as f:
        knowledge_base = json.load(f)
    for chunk in knowledge_base["chunks"]:
        chunk["embedding"] = np.array(chunk["embedding"])
    print(str(knowledge_base['total_chunks']) + " chunks carregados")
    print("Livros: " + ', '.join(knowledge_base['books']))


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


def get_query_embedding(query):
    if GEMINI_API_KEY and GEMINI_API_KEY != "cole_sua_chave_aqui":
        try:
            url = "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key=" + GEMINI_API_KEY
            payload = {
                "model": "models/text-embedding-004",
                "content": {"parts": [{"text": query}]}
            }
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code == 200:
                return np.array(resp.json()["embedding"]["values"])
        except Exception as e:
            print("Gemini embedding falhou: " + str(e))
    try:
        from sentence_transformers import SentenceTransformer
        if not hasattr(get_query_embedding, '_model'):
            model_name = knowledge_base.get("model", "all-MiniLM-L6-v2")
            get_query_embedding._model = SentenceTransformer(model_name)
        return np.array(get_query_embedding._model.encode(query))
    except ImportError:
        return None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "").strip()
    history = data.get("history", [])
    if not question:
        return jsonify({"error": "Pergunta vazia"}), 400
    start = time.time()
    query_emb = get_query_embedding(question)
    if query_emb is None:
        return jsonify({
            "answer": "Nao foi possivel gerar embedding. Verifique a configuracao.",
            "sources": [], "time": 0, "provider": "none"
        })
    results = search_similar_chunks(query_emb, top_k=TOP_K)
    context_parts = []
    for i, chunk in enumerate(results, 1):
        context_parts.append("[Trecho " + str(i) + " - " + chunk['book'] + ", p." + str(chunk['page']) + "]\n" + chunk['text'])
    context = "\n\n---\n\n".join(context_parts)[:4000]
    system_prompt = SYSTEM_PROMPT + context[:1500]
    messages = []
    if history:
        for msg in history[-MAX_HISTORY:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": question})
    response = api_manager.generate(system_prompt, messages)
    elapsed = time.time() - start
    sources = []
    seen = set()
    for r in results:
        key = r['book'] + "-p" + str(r['page'])
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


load_knowledge_base()

if __name__ == "__main__":
    print("MEDIC.RG v2 - Servidor: http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
