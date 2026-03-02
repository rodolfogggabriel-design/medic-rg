import os
import json
import numpy as np
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import requests
import time
import re
from collections import Counter

from multi_api import MultiAPIManager

load_dotenv()

app = Flask(__name__)

KNOWLEDGE_BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'knowledge_base.json')
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
TOP_K = 2
MAX_HISTORY = 10

knowledge_base = None
api_manager = MultiAPIManager()

SYSTEM_PROMPT = """Voce e o MEDIC.RG, um assistente de estudos medicos especializado.
Voce responde perguntas com base nos livros de medicina fornecidos como contexto.

REGRAS:
1. Responda SEMPRE em portugues brasileiro
2. Base suas respostas EXCLUSIVAMENTE no contexto fornecido dos livros
3. Cite o livro e a pagina quando referenciar informacoes especificas (ex: "Segundo Guyton, p.45...")
4. Se o contexto nao contiver informacao suficiente, diga claramente
5. Use linguagem acessivel mas tecnicamente precisa
6. Organize a resposta com clareza (use paragrafos, nao listas excessivas)
7. Se relevante, faca conexoes entre conceitos de diferentes livros
8. Ao final, sugira topicos relacionados que o aluno pode estudar

CONTEXTO DOS LIVROS:
"""

FLASHCARD_PROMPT = """Com base EXCLUSIVAMENTE no contexto dos livros fornecido abaixo, gere exatamente 5 flashcards sobre o tema solicitado.

FORMATO OBRIGATORIO (responda APENAS com este JSON, sem texto antes ou depois):
[
  {"frente": "pergunta clara e objetiva", "verso": "resposta concisa baseada nos livros", "livro": "nome do livro", "pagina": "numero"},
  {"frente": "...", "verso": "...", "livro": "...", "pagina": "..."}
]

REGRAS:
- Use APENAS informacoes presentes no contexto
- Perguntas devem testar conhecimento medico real
- Respostas devem ser concisas mas completas
- Cite o livro e pagina de onde veio a informacao
- Responda em portugues brasileiro

CONTEXTO DOS LIVROS:
"""

QUIZ_PROMPT = """Com base EXCLUSIVAMENTE no contexto dos livros fornecido abaixo, gere exatamente 5 questoes de multipla escolha sobre o tema solicitado.

FORMATO OBRIGATORIO (responda APENAS com este JSON, sem texto antes ou depois):
[
  {
    "pergunta": "texto da pergunta",
    "alternativas": ["a) ...", "b) ...", "c) ...", "d) ..."],
    "correta": 0,
    "explicacao": "explicacao da resposta correta citando o livro",
    "livro": "nome do livro",
    "pagina": "numero"
  }
]

REGRAS:
- Use APENAS informacoes presentes no contexto
- Questoes devem ter 4 alternativas (a, b, c, d)
- "correta" e o indice da alternativa correta (0=a, 1=b, 2=c, 3=d)
- Alternativas incorretas devem ser plausíveis mas claramente erradas
- Explicacao deve citar o livro e pagina
- Responda em portugues brasileiro

CONTEXTO DOS LIVROS:
"""

MINDMAP_PROMPT = """Com base EXCLUSIVAMENTE no contexto dos livros fornecido abaixo, gere um mapa mental sobre o tema solicitado.

FORMATO OBRIGATORIO (responda APENAS com este JSON, sem texto antes ou depois):
{
  "centro": "tema principal",
  "ramos": [
    {
      "titulo": "subtema 1",
      "cor": "#00e5c8",
      "itens": ["conceito A", "conceito B", "conceito C"]
    },
    {
      "titulo": "subtema 2",
      "cor": "#a78bfa",
      "itens": ["conceito D", "conceito E"]
    },
    {
      "titulo": "subtema 3",
      "cor": "#ff6b9d",
      "itens": ["conceito F", "conceito G", "conceito H"]
    },
    {
      "titulo": "subtema 4",
      "cor": "#60a5fa",
      "itens": ["conceito I", "conceito J"]
    }
  ],
  "livros": ["livro 1, p.X", "livro 2, p.Y"]
}

REGRAS:
- Gere entre 3 e 6 ramos principais
- Cada ramo deve ter entre 2 e 5 itens
- Use cores variadas: #00e5c8, #a78bfa, #ff6b9d, #60a5fa, #fbbf24, #34d399
- Organize hierarquicamente do geral pro especifico
- Use APENAS informacoes presentes no contexto
- Responda em portugues brasileiro

CONTEXTO DOS LIVROS:
"""


def load_knowledge_base():
    global knowledge_base
    if not os.path.exists(KNOWLEDGE_BASE_PATH):
        print("Base de conhecimento nao encontrada!")
        print("Caminho: " + KNOWLEDGE_BASE_PATH)
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


def keyword_search(query, top_k=TOP_K):
    if not knowledge_base or not knowledge_base["chunks"]:
        return []

    stopwords = {
        'o', 'a', 'os', 'as', 'um', 'uma', 'uns', 'umas', 'de', 'do', 'da',
        'dos', 'das', 'em', 'no', 'na', 'nos', 'nas', 'por', 'para', 'com',
        'sem', 'que', 'qual', 'quais', 'como', 'onde', 'quando', 'se', 'e',
        'ou', 'mas', 'pois', 'porque', 'entre', 'sobre', 'mais', 'menos',
        'muito', 'pouco', 'todo', 'toda', 'ser', 'ter', 'estar', 'fazer',
        'pode', 'tem', 'sao', 'foi', 'eh', 'isso', 'este', 'esta', 'esse',
        'essa', 'ao', 'pela', 'pelo', 'num', 'numa', 'the', 'is', 'of',
        'and', 'to', 'in', 'it', 'for', 'on', 'are', 'was', 'what', 'how',
    }

    query_lower = query.lower()
    query_words = re.findall(r'\w+', query_lower)
    query_words = [w for w in query_words if w not in stopwords and len(w) > 2]

    if not query_words:
        query_words = re.findall(r'\w+', query_lower)

    scored = []
    for chunk in knowledge_base["chunks"]:
        text_lower = chunk["text"].lower()
        score = 0
        for word in query_words:
            count = text_lower.count(word)
            score += count
        if query_lower in text_lower:
            score += 10
        for i in range(len(query_words) - 1):
            bigram = query_words[i] + " " + query_words[i + 1]
            if bigram in text_lower:
                score += 5
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    max_score = scored[0][0] if scored else 1
    for score, chunk in scored[:top_k]:
        results.append({
            "text": chunk["text"],
            "book": chunk["book"],
            "page": chunk["page"],
            "similarity": round(min(score / max(max_score, 1), 1.0) * 100) / 100
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
            else:
                print("Gemini embedding status: " + str(resp.status_code))
        except Exception as e:
            print("Gemini embedding falhou: " + str(e))
    return None


def get_context_for_topic(topic, top_k=4):
    """Busca contexto relevante para um topico (usado por flashcards e quiz)."""
    query_emb = get_query_embedding(topic)
    if query_emb is not None:
        results = search_similar_chunks(query_emb, top_k=top_k)
    else:
        results = keyword_search(topic, top_k=top_k)

    if not results:
        return "", []

    context_parts = []
    for i, chunk in enumerate(results, 1):
        context_parts.append("[Trecho " + str(i) + " - " + chunk['book'] + ", p." + str(chunk['page']) + "]\n" + chunk['text'])
    context = "\n\n---\n\n".join(context_parts)[:3000]

    sources = []
    seen = set()
    for r in results:
        key = r['book'] + "-p" + str(r['page'])
        if key not in seen:
            sources.append({"book": r["book"], "page": r["page"]})
            seen.add(key)
    return context, sources


def parse_json_response(text):
    """Tenta extrair JSON de uma resposta da API."""
    text = text.strip()
    # Remove markdown code blocks
    text = re.sub(r'^```json\s*', '', text)
    text = re.sub(r'^```\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Tenta encontrar array JSON no texto
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
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

    query_emb = get_query_embedding(question)
    if query_emb is not None:
        results = search_similar_chunks(query_emb, top_k=TOP_K)
    else:
        results = keyword_search(question, top_k=TOP_K)

    if not results:
        return jsonify({
            "answer": "Nao encontrei informacoes relevantes nos livros para essa pergunta. Tente reformular.",
            "sources": [], "time": 0, "provider": "none"
        })

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


@app.route("/api/flashcards", methods=["POST"])
def flashcards():
    data = request.json
    topic = data.get("topic", "").strip()

    if not topic:
        return jsonify({"error": "Tema vazio"}), 400

    start = time.time()

    context, sources = get_context_for_topic(topic, top_k=4)
    if not context:
        return jsonify({
            "error": "Nao encontrei conteudo sobre esse tema nos livros.",
            "cards": []
        })

    system_prompt = FLASHCARD_PROMPT + context[:2000]
    messages = [{"role": "user", "content": "Gere 5 flashcards sobre: " + topic}]

    response = api_manager.generate(system_prompt, messages)

    cards = parse_json_response(response["text"])
    if cards is None:
        cards = []

    elapsed = time.time() - start

    return jsonify({
        "cards": cards,
        "sources": sources,
        "time": round(elapsed, 2),
        "provider": response["provider"]
    })


@app.route("/api/quiz", methods=["POST"])
def quiz():
    data = request.json
    topic = data.get("topic", "").strip()

    if not topic:
        return jsonify({"error": "Tema vazio"}), 400

    start = time.time()

    context, sources = get_context_for_topic(topic, top_k=4)
    if not context:
        return jsonify({
            "error": "Nao encontrei conteudo sobre esse tema nos livros.",
            "questions": []
        })

    system_prompt = QUIZ_PROMPT + context[:2000]
    messages = [{"role": "user", "content": "Gere 5 questoes de multipla escolha sobre: " + topic}]

    response = api_manager.generate(system_prompt, messages)

    questions = parse_json_response(response["text"])
    if questions is None:
        questions = []

    elapsed = time.time() - start

    return jsonify({
        "questions": questions,
        "sources": sources,
        "time": round(elapsed, 2),
        "provider": response["provider"]
    })


@app.route("/api/mindmap", methods=["POST"])
def mindmap():
    data = request.json
    topic = data.get("topic", "").strip()

    if not topic:
        return jsonify({"error": "Tema vazio"}), 400

    start = time.time()

    context, sources = get_context_for_topic(topic, top_k=4)
    if not context:
        return jsonify({
            "error": "Nao encontrei conteudo sobre esse tema nos livros.",
            "mindmap": None
        })

    system_prompt = MINDMAP_PROMPT + context[:2000]
    messages = [{"role": "user", "content": "Gere um mapa mental sobre: " + topic}]

    response = api_manager.generate(system_prompt, messages)

    mindmap_data = parse_json_response(response["text"])

    elapsed = time.time() - start

    return jsonify({
        "mindmap": mindmap_data,
        "sources": sources,
        "time": round(elapsed, 2),
        "provider": response["provider"]
    })


@app.route("/api/topics")
def topics():
    """Retorna lista de topicos disponiveis baseado nos livros."""
    if not knowledge_base or not knowledge_base["chunks"]:
        return jsonify({"topics": []})

    topic_counts = Counter()
    for chunk in knowledge_base["chunks"]:
        book = chunk.get("book", "")
        if book:
            topic_counts[book] += 1

    books_info = []
    for book, count in topic_counts.most_common():
        books_info.append({"name": book, "chunks": count})

    return jsonify({"books": books_info, "total_chunks": knowledge_base.get("total_chunks", 0)})


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
    print("=" * 60)
    print("  MEDIC.RG v2 - Multi-API Free Tier Rotation")
    print("=" * 60)
    print("Servidor: http://localhost:5000")
    print("Ctrl+C para parar")
    app.run(host="0.0.0.0", port=5000, debug=True)
