import os
import json
import numpy as np
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import requests
import time
import re
import uuid
import hashlib
from collections import Counter

from multi_api import MultiAPIManager

load_dotenv()

app = Flask(__name__)

KNOWLEDGE_BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'knowledge_base.json')
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")  # anon key
TOP_K = 2
MAX_HISTORY = 10

knowledge_base = None
api_manager = MultiAPIManager()

# =============================================
# PROMPTS
# =============================================
SYSTEM_PROMPT = """Voce e o MEDIC.RG, um assistente de estudos medicos especializado.
Voce responde perguntas com base nos livros de medicina fornecidos como contexto.

REGRAS:
1. Responda SEMPRE em portugues brasileiro
2. Base suas respostas EXCLUSIVAMENTE no contexto fornecido dos livros
3. Cite o livro e a pagina quando referenciar informacoes especificas
4. Se o contexto nao contiver informacao suficiente, diga claramente
5. Use linguagem acessivel mas tecnicamente precisa
6. Organize a resposta com clareza
7. Se relevante, faca conexoes entre conceitos de diferentes livros
8. Ao final, sugira topicos relacionados que o aluno pode estudar

CONTEXTO DOS LIVROS:
"""

FLASHCARD_PROMPT = """Com base EXCLUSIVAMENTE no contexto dos livros fornecido abaixo, gere exatamente 5 flashcards sobre o tema solicitado.

FORMATO OBRIGATORIO - responda APENAS com este JSON, sem texto antes ou depois, sem markdown:
[
  {"frente": "pergunta clara e objetiva", "verso": "resposta concisa baseada nos livros", "livro": "nome do livro", "pagina": "numero"},
  {"frente": "pergunta 2", "verso": "resposta 2", "livro": "nome", "pagina": "numero"},
  {"frente": "pergunta 3", "verso": "resposta 3", "livro": "nome", "pagina": "numero"},
  {"frente": "pergunta 4", "verso": "resposta 4", "livro": "nome", "pagina": "numero"},
  {"frente": "pergunta 5", "verso": "resposta 5", "livro": "nome", "pagina": "numero"}
]

REGRAS:
- Gere exatamente 5 flashcards, nem mais nem menos
- Use APENAS informacoes presentes no contexto
- Responda em portugues brasileiro
- NAO inclua texto antes ou depois do JSON
- NAO use markdown ou code blocks

CONTEXTO DOS LIVROS:
"""

QUIZ_PROMPT = """Com base EXCLUSIVAMENTE no contexto dos livros fornecido abaixo, gere exatamente 5 questoes de multipla escolha sobre o tema solicitado.

FORMATO OBRIGATORIO - responda APENAS com este JSON, sem texto antes ou depois, sem markdown:
[
  {
    "pergunta": "texto da pergunta",
    "alternativas": ["a) opcao 1", "b) opcao 2", "c) opcao 3", "d) opcao 4"],
    "correta": 0,
    "explicacao": "explicacao da resposta correta citando o livro",
    "livro": "nome do livro",
    "pagina": "numero"
  }
]

REGRAS:
- Gere exatamente 5 questoes, nem mais nem menos
- Cada questao deve ter 4 alternativas (a, b, c, d)
- "correta" e o indice da alternativa correta (0=a, 1=b, 2=c, 3=d)
- Use APENAS informacoes presentes no contexto
- Responda em portugues brasileiro
- NAO inclua texto antes ou depois do JSON

CONTEXTO DOS LIVROS:
"""

MINDMAP_PROMPT = """Com base EXCLUSIVAMENTE no contexto dos livros, gere um mapa mental sobre o tema solicitado.
FORMATO OBRIGATORIO (responda APENAS com este JSON):
{"centro":"tema","resumo":"resumo de 2-3 frases do tema baseado nos livros","ramos":[{"titulo":"subtema","cor":"#00e5c8","itens":["conceito A","conceito B"]}],"livros":["livro, p.X"]}
Use 3-6 ramos, 2-5 itens cada. Cores: #00e5c8, #a78bfa, #ff6b9d, #60a5fa, #fbbf24, #34d399. Portugues brasileiro. CONTEXTO:
"""

EXAM_MC_PROMPT = """Gere {n} questoes de MULTIPLA ESCOLHA nivel {nivel} sobre "{tema}".
FORMATO JSON (sem texto extra):
[{{"pergunta":"texto","alternativas":["a)...","b)...","c)...","d)..."],"correta":0,"explicacao":"explicacao","peso":{peso},"livro":"nome","pagina":"num","saiba_mais":"topico para aprofundar","sugestao_chat":"pergunta sugerida pro chat"}}]
Nivel facil=conceitos basicos, medio=aplicacao clinica, dificil=raciocinio complexo. Portugues brasileiro.
CONTEXTO:
"""

EXAM_DISC_PROMPT = """Gere {n} questoes DISCURSIVAS nivel {nivel} sobre "{tema}".
FORMATO JSON (sem texto extra):
[{{"pergunta":"texto da questao discursiva","resposta_esperada":"resposta modelo completa","pontos_chave":["ponto 1","ponto 2","ponto 3"],"peso":{peso},"livro":"nome","pagina":"num","saiba_mais":"topico para aprofundar","sugestao_chat":"pergunta sugerida pro chat"}}]
Portugues brasileiro.
CONTEXTO:
"""

EXAM_MATCH_PROMPT = """Gere {n} questoes de LIGAR COLUNAS nivel {nivel} sobre "{tema}".
Cada questao tem uma coluna A e coluna B que devem ser conectadas.
FORMATO JSON (sem texto extra):
[{{"instrucao":"Ligue os itens da coluna A com a coluna B","coluna_a":["item 1","item 2","item 3","item 4"],"coluna_b":["par 1","par 2","par 3","par 4"],"pares_corretos":[0,1,2,3],"peso":{peso},"explicacao":"explicacao","livro":"nome","pagina":"num","saiba_mais":"topico","sugestao_chat":"pergunta"}}]
pares_corretos: indice da coluna_b que corresponde a cada item da coluna_a.
Portugues brasileiro.
CONTEXTO:
"""

EXAM_ORDER_PROMPT = """Gere {n} questoes de ORDENAR SEQUENCIA nivel {nivel} sobre "{tema}".
O aluno deve colocar os itens na ordem correta.
FORMATO JSON (sem texto extra):
[{{"instrucao":"Ordene os passos/eventos na sequencia correta","itens_embaralhados":["passo C","passo A","passo D","passo B"],"ordem_correta":[1,3,0,2],"peso":{peso},"explicacao":"explicacao da ordem correta","livro":"nome","pagina":"num","saiba_mais":"topico","sugestao_chat":"pergunta"}}]
ordem_correta: indice que cada item embaralhado deve ocupar (0=primeiro, 1=segundo...).
Portugues brasileiro.
CONTEXTO:
"""

STUDY_GUIDE_PROMPT = """Com base EXCLUSIVAMENTE no contexto dos livros, gere um GUIA DE ESTUDO completo sobre o tema.
FORMATO JSON (sem texto extra):
{{"titulo":"Guia: tema","resumo":"resumo executivo de 3-4 frases","topicos":[{{"titulo":"topico 1","conteudo":"explicacao detalhada","importancia":"alta/media/baixa"}}],"termos_chave":[{{"termo":"palavra","definicao":"definicao curta"}}],"dicas_estudo":["dica 1","dica 2"],"perguntas_frequentes":["pergunta 1","pergunta 2","pergunta 3"],"livros_referencia":["livro, p.X"]}}
Portugues brasileiro.
CONTEXTO:
"""


# =============================================
# KNOWLEDGE BASE
# =============================================
def load_knowledge_base():
    global knowledge_base
    if not os.path.exists(KNOWLEDGE_BASE_PATH):
        print("Base de conhecimento nao encontrada: " + KNOWLEDGE_BASE_PATH)
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
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0
    return dot / (na * nb)


def search_similar_chunks(query_embedding, top_k=TOP_K, chunks=None):
    target = chunks if chunks else (knowledge_base.get("chunks", []) if knowledge_base else [])
    if not target:
        return []
    sims = []
    for chunk in target:
        emb = chunk.get("embedding")
        if emb is None:
            continue
        if not isinstance(emb, np.ndarray):
            emb = np.array(emb)
        sim = cosine_similarity(query_embedding, emb)
        sims.append((sim, chunk))
    sims.sort(key=lambda x: x[0], reverse=True)
    results = []
    for sim, chunk in sims[:top_k]:
        results.append({"text": chunk["text"], "book": chunk.get("book", "Material"), "page": chunk.get("page", ""), "similarity": float(sim)})
    return results


def keyword_search(query, top_k=TOP_K, chunks=None):
    target = chunks if chunks else (knowledge_base.get("chunks", []) if knowledge_base else [])
    if not target:
        return []
    stopwords = {'o','a','os','as','um','uma','de','do','da','dos','das','em','no','na','nos','nas','por','para','com','sem','que','qual','como','onde','quando','se','e','ou','mas','pois','porque','entre','sobre','mais','menos','muito','pouco','todo','toda','ser','ter','estar','fazer','pode','tem','sao','foi','eh','isso','este','esta','esse','essa','ao','pela','pelo'}
    qw = [w for w in re.findall(r'\w+', query.lower()) if w not in stopwords and len(w) > 2]
    if not qw:
        qw = re.findall(r'\w+', query.lower())
    scored = []
    for chunk in target:
        tl = chunk["text"].lower()
        score = sum(tl.count(w) for w in qw)
        if query.lower() in tl:
            score += 10
        for i in range(len(qw)-1):
            if qw[i] + " " + qw[i+1] in tl:
                score += 5
        if score > 0:
            scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    mx = scored[0][0] if scored else 1
    return [{"text": c["text"], "book": c.get("book","Material"), "page": c.get("page",""), "similarity": round(min(s/max(mx,1),1.0)*100)/100} for s, c in scored[:top_k]]


def get_query_embedding(query):
    if GEMINI_API_KEY and GEMINI_API_KEY != "cole_sua_chave_aqui":
        try:
            url = "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key=" + GEMINI_API_KEY
            resp = requests.post(url, json={"model":"models/text-embedding-004","content":{"parts":[{"text":query}]}}, timeout=10)
            if resp.status_code == 200:
                return np.array(resp.json()["embedding"]["values"])
        except:
            pass
    return None


def get_context_for_topic(topic, top_k=4, chunks=None):
    query_emb = get_query_embedding(topic)
    if query_emb is not None:
        results = search_similar_chunks(query_emb, top_k=top_k, chunks=chunks)
    else:
        results = keyword_search(topic, top_k=top_k, chunks=chunks)
    if not results:
        return "", []
    parts = []
    for i, c in enumerate(results, 1):
        parts.append("[Trecho " + str(i) + " - " + c['book'] + ", p." + str(c['page']) + "]\n" + c['text'])
    context = "\n\n---\n\n".join(parts)[:3000]
    seen = set()
    sources = []
    for r in results:
        key = r['book'] + "-p" + str(r['page'])
        if key not in seen:
            sources.append({"book": r["book"], "page": r["page"]})
            seen.add(key)
    return context, sources


def parse_json_response(text):
    if not text:
        return None
    text = text.strip()
    # Remove markdown code blocks
    text = re.sub(r'^```json\s*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'^```\s*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n?\s*```\s*$', '', text, flags=re.MULTILINE)
    text = text.strip()
    # Try direct parse
    try:
        return json.loads(text)
    except:
        pass
    # Try to find JSON array
    match = re.search(r'\[[\s\S]*\]', text)
    if match:
        try:
            return json.loads(match.group())
        except:
            # Try fixing common issues: trailing commas
            fixed = re.sub(r',\s*([}\]])', r'\1', match.group())
            try:
                return json.loads(fixed)
            except:
                pass
    # Try to find JSON object
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except:
            fixed = re.sub(r',\s*([}\]])', r'\1', match.group())
            try:
                return json.loads(fixed)
            except:
                pass
    # Last resort: try to find multiple JSON objects and wrap in array
    objects = re.findall(r'\{[^{}]+\}', text)
    if objects:
        try:
            parsed = [json.loads(o) for o in objects]
            return parsed
        except:
            pass
    print("PARSE FAILED for text: " + text[:200])
    return None


# =============================================
# SUPABASE HELPERS
# =============================================
def supabase_request(method, path, data=None, token=None):
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    headers = {
        "apikey": SUPABASE_KEY,
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }
    if token:
        headers["Authorization"] = "Bearer " + token
    url = SUPABASE_URL + "/rest/v1/" + path
    try:
        if method == "GET":
            r = requests.get(url, headers=headers, timeout=10)
        elif method == "POST":
            r = requests.post(url, headers=headers, json=data, timeout=10)
        elif method == "DELETE":
            r = requests.delete(url, headers=headers, timeout=10)
        elif method == "PATCH":
            r = requests.patch(url, headers=headers, json=data, timeout=10)
        else:
            return None
        if r.status_code < 300:
            return r.json() if r.text else {}
    except:
        pass
    return None


def get_user_from_token(req):
    auth = req.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None, None
    token = auth[7:]
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None, token
    try:
        r = requests.get(SUPABASE_URL + "/auth/v1/user", headers={"apikey": SUPABASE_KEY, "Authorization": "Bearer " + token}, timeout=10)
        if r.status_code == 200:
            return r.json(), token
    except:
        pass
    return None, token


# =============================================
# ROUTES
# =============================================
@app.route("/")
def index():
    books = knowledge_base.get("books", []) if knowledge_base else []
    total = knowledge_base.get("total_chunks", 0) if knowledge_base else 0
    return render_template("index.html", books=books, total_chunks=total,
                         supabase_url=SUPABASE_URL, supabase_key=SUPABASE_KEY)


@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "").strip()
    history = data.get("history", [])
    material_id = data.get("material_id")

    if not question:
        return jsonify({"error": "Pergunta vazia"}), 400

    start = time.time()

    # Use custom material chunks if specified
    chunks = None
    if material_id:
        user, token = get_user_from_token(request)
        if user and token:
            mat = supabase_request("GET", "materials?id=eq." + material_id + "&user_id=eq." + user["id"], token=token)
            if mat and len(mat) > 0:
                chunks = mat[0].get("chunks", [])

    query_emb = get_query_embedding(question)
    if query_emb is not None:
        results = search_similar_chunks(query_emb, top_k=TOP_K, chunks=chunks)
    else:
        results = keyword_search(question, top_k=TOP_K, chunks=chunks)

    if not results:
        return jsonify({"answer": "Nao encontrei informacoes relevantes. Tente reformular.", "sources": [], "time": 0, "provider": "none"})

    context_parts = []
    for i, c in enumerate(results, 1):
        context_parts.append("[Trecho " + str(i) + " - " + c['book'] + ", p." + str(c['page']) + "]\n" + c['text'])
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
            sources.append({"book": r["book"], "page": r["page"], "relevance": round(r["similarity"]*100, 1)})
            seen.add(key)

    return jsonify({"answer": response["text"], "sources": sources, "time": round(elapsed, 2), "provider": response["provider"], "fallback": response["fallback"]})


@app.route("/api/flashcards", methods=["POST"])
def flashcards():
    data = request.json
    topic = data.get("topic", "").strip()
    if not topic:
        return jsonify({"error": "Tema vazio"}), 400
    start = time.time()
    context, sources = get_context_for_topic(topic, top_k=4)
    if not context:
        return jsonify({"error": "Tema nao encontrado.", "cards": []})
    system_prompt = FLASHCARD_PROMPT + context[:2000]
    response = api_manager.generate(system_prompt, [{"role": "user", "content": "Gere exatamente 5 flashcards sobre: " + topic + "\n\nResponda APENAS com o JSON, sem texto adicional."}])
    raw = response["text"]
    cards = parse_json_response(raw)
    if cards is None:
        print("FLASHCARD RAW RESPONSE: " + raw[:500])
        cards = []
    if isinstance(cards, dict):
        cards = cards.get("flashcards", cards.get("cards", [cards]))
    if not isinstance(cards, list):
        cards = []
    return jsonify({"cards": cards, "sources": sources, "time": round(time.time()-start, 2), "provider": response["provider"]})


@app.route("/api/quiz", methods=["POST"])
def quiz():
    data = request.json
    topic = data.get("topic", "").strip()
    if not topic:
        return jsonify({"error": "Tema vazio"}), 400
    start = time.time()
    context, sources = get_context_for_topic(topic, top_k=4)
    if not context:
        return jsonify({"error": "Tema nao encontrado.", "questions": []})
    system_prompt = QUIZ_PROMPT + context[:2000]
    response = api_manager.generate(system_prompt, [{"role": "user", "content": "Gere exatamente 5 questoes de multipla escolha sobre: " + topic + "\n\nResponda APENAS com o JSON, sem texto adicional."}])
    raw = response["text"]
    questions = parse_json_response(raw)
    if questions is None:
        print("QUIZ RAW RESPONSE: " + raw[:500])
        questions = []
    if isinstance(questions, dict):
        questions = questions.get("questions", questions.get("questoes", [questions]))
    if not isinstance(questions, list):
        questions = []
    return jsonify({"questions": questions, "sources": sources, "time": round(time.time()-start, 2), "provider": response["provider"]})


@app.route("/api/mindmap", methods=["POST"])
def mindmap():
    data = request.json
    topic = data.get("topic", "").strip()
    if not topic:
        return jsonify({"error": "Tema vazio"}), 400
    start = time.time()
    context, sources = get_context_for_topic(topic, top_k=4)
    if not context:
        return jsonify({"error": "Tema nao encontrado.", "mindmap": None})
    system_prompt = MINDMAP_PROMPT + context[:2000]
    response = api_manager.generate(system_prompt, [{"role": "user", "content": "Gere mapa mental sobre: " + topic}])
    mm = parse_json_response(response["text"])
    return jsonify({"mindmap": mm, "sources": sources, "time": round(time.time()-start, 2), "provider": response["provider"]})


@app.route("/api/exam", methods=["POST"])
def exam():
    data = request.json
    topic = data.get("topic", "").strip()
    types = data.get("types", ["mc"])  # mc, disc, match, order
    difficulty = data.get("difficulty", "medio")
    n_per_type = data.get("n_per_type", 3)
    weight = data.get("weight", 1)

    if not topic:
        return jsonify({"error": "Tema vazio"}), 400

    start = time.time()
    context, sources = get_context_for_topic(topic, top_k=6)
    if not context:
        return jsonify({"error": "Tema nao encontrado.", "questions": []})

    all_questions = []

    for qtype in types:
        if qtype == "mc":
            prompt_tmpl = EXAM_MC_PROMPT
        elif qtype == "disc":
            prompt_tmpl = EXAM_DISC_PROMPT
        elif qtype == "match":
            prompt_tmpl = EXAM_MATCH_PROMPT
        elif qtype == "order":
            prompt_tmpl = EXAM_ORDER_PROMPT
        else:
            continue

        prompt = prompt_tmpl.format(n=n_per_type, nivel=difficulty, tema=topic, peso=weight)
        system_prompt = prompt + context[:2000]
        response = api_manager.generate(system_prompt, [{"role": "user", "content": "Gere as questoes."}])
        parsed = parse_json_response(response["text"])
        if parsed and isinstance(parsed, list):
            for q in parsed:
                q["type"] = qtype
            all_questions.extend(parsed)

    return jsonify({
        "questions": all_questions,
        "sources": sources,
        "time": round(time.time()-start, 2),
        "topic": topic,
        "difficulty": difficulty
    })


@app.route("/api/study-guide", methods=["POST"])
def study_guide():
    data = request.json
    topic = data.get("topic", "").strip()
    if not topic:
        return jsonify({"error": "Tema vazio"}), 400
    start = time.time()
    context, sources = get_context_for_topic(topic, top_k=6)
    if not context:
        return jsonify({"error": "Tema nao encontrado."})
    system_prompt = STUDY_GUIDE_PROMPT + context[:2500]
    response = api_manager.generate(system_prompt, [{"role": "user", "content": "Gere guia de estudo sobre: " + topic}])
    guide = parse_json_response(response["text"])
    return jsonify({"guide": guide, "sources": sources, "time": round(time.time()-start, 2), "provider": response["provider"]})


@app.route("/api/progress", methods=["POST"])
def save_progress():
    user, token = get_user_from_token(request)
    if not user:
        return jsonify({"error": "Nao autenticado"}), 401
    data = request.json
    result = supabase_request("POST", "progress", {
        "user_id": user["id"],
        "type": data.get("type", "quiz"),
        "topic": data.get("topic", ""),
        "score": data.get("score", 0),
        "max_score": data.get("max_score", 0),
        "difficulty": data.get("difficulty", "medio"),
        "details": data.get("details", {})
    }, token=token)
    if result:
        return jsonify({"ok": True})
    return jsonify({"error": "Erro ao salvar"}), 500


@app.route("/api/progress", methods=["GET"])
def get_progress():
    user, token = get_user_from_token(request)
    if not user:
        return jsonify({"error": "Nao autenticado"}), 401
    data = supabase_request("GET", "progress?user_id=eq." + user["id"] + "&order=created_at.desc&limit=50", token=token)
    return jsonify({"progress": data or []})


@app.route("/api/upload-material", methods=["POST"])
def upload_material():
    user, token = get_user_from_token(request)
    if not user:
        return jsonify({"error": "Nao autenticado"}), 401

    data = request.json
    name = data.get("name", "Material sem nome")
    text = data.get("text", "")

    if not text or len(text) < 50:
        return jsonify({"error": "Texto muito curto (minimo 50 caracteres)"}), 400

    # Chunk the text
    chunk_size = 800
    overlap = 100
    chunks = []
    words = text.split()
    i = 0
    while i < len(words):
        chunk_words = words[i:i+chunk_size]
        chunk_text = " ".join(chunk_words)

        # Generate embedding
        emb = get_query_embedding(chunk_text[:500])
        embedding_list = emb.tolist() if emb is not None else []

        chunks.append({
            "text": chunk_text,
            "book": name,
            "page": str(len(chunks)+1),
            "embedding": embedding_list
        })
        i += chunk_size - overlap

    result = supabase_request("POST", "materials", {
        "user_id": user["id"],
        "name": name,
        "description": text[:200],
        "chunks": chunks,
        "total_chunks": len(chunks)
    }, token=token)

    if result:
        return jsonify({"ok": True, "id": result[0]["id"] if isinstance(result, list) else "", "chunks": len(chunks)})
    return jsonify({"error": "Erro ao salvar material"}), 500


@app.route("/api/materials", methods=["GET"])
def get_materials():
    user, token = get_user_from_token(request)
    if not user:
        return jsonify({"error": "Nao autenticado"}), 401
    data = supabase_request("GET", "materials?user_id=eq." + user["id"] + "&select=id,name,description,total_chunks,created_at&order=created_at.desc", token=token)
    return jsonify({"materials": data or []})


@app.route("/api/materials/<mat_id>", methods=["DELETE"])
def delete_material(mat_id):
    user, token = get_user_from_token(request)
    if not user:
        return jsonify({"error": "Nao autenticado"}), 401
    supabase_request("DELETE", "materials?id=eq." + mat_id + "&user_id=eq." + user["id"], token=token)
    return jsonify({"ok": True})


@app.route("/api/topics")
def topics():
    if not knowledge_base or not knowledge_base["chunks"]:
        return jsonify({"topics": []})
    topic_counts = Counter()
    for chunk in knowledge_base["chunks"]:
        book = chunk.get("book", "")
        if book:
            topic_counts[book] += 1
    books_info = [{"name": b, "chunks": c} for b, c in topic_counts.most_common()]
    return jsonify({"books": books_info, "total_chunks": knowledge_base.get("total_chunks", 0)})


@app.route("/api/status")
def status():
    api_status = api_manager.get_status()
    kb_status = "empty" if (not knowledge_base or not knowledge_base["chunks"]) else "ready"
    return jsonify({
        "status": kb_status,
        "books": knowledge_base.get("books", []) if knowledge_base else [],
        "total_chunks": knowledge_base.get("total_chunks", 0) if knowledge_base else 0,
        "apis": api_status,
        "supabase": bool(SUPABASE_URL and SUPABASE_KEY)
    })


@app.route("/api/providers")
def providers():
    return jsonify(api_manager.get_status())


load_knowledge_base()

if __name__ == "__main__":
    print("=" * 60)
    print("  MEDIC.RG v4 - Plataforma de Estudos Medicos")
    print("=" * 60)
    print("Servidor: http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
