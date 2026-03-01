# 🏥 MedicQA v2 — Assistente de Estudos Médicos (Multi-API)

Sistema de perguntas e respostas para estudantes de medicina, com respostas baseadas nos livros **Guyton**, **Silverthorn** e **Porto**.

**Novidade v2:** Combina **7 APIs gratuitas** com rotação automática. Quando uma atinge o limite, pula pra próxima. Resultado: uso praticamente **ilimitado**, custo **ZERO**.

---

## 📡 APIs Suportadas (todas grátis)

| # | API | Limite Free | Modelo | Qualidade |
|---|-----|------------|--------|-----------|
| 1 | **Google Gemini** | 15 req/min, 1500/dia | Gemini 2.0 Flash | ⭐⭐⭐⭐⭐ |
| 2 | **Groq** | 30 req/min, 14400/dia | Llama 3.3 70B | ⭐⭐⭐⭐ |
| 3 | **Mistral** | 60 req/min | Mistral Small | ⭐⭐⭐⭐ |
| 4 | **Cohere** | 20 req/min, 1000/mês | Command-R | ⭐⭐⭐ |
| 5 | **HuggingFace** | 1000 req/dia | Mistral 7B | ⭐⭐⭐ |
| 6 | **Together AI** | $5 crédito grátis | Llama 3.3 70B | ⭐⭐⭐⭐ |
| 7 | **OpenRouter** | Modelos grátis | Llama 3.3 8B | ⭐⭐⭐ |

**Combinando todas:** ~140+ req/min, ~17.000+ req/dia = praticamente ilimitado para 1 pessoa.

---

## 🚀 Setup Rápido

### Passo 1: Pegar as chaves (5-10 min)

Acesse cada link e crie uma chave gratuita:

1. **Gemini** (obrigatório): https://aistudio.google.com/apikey
2. **Groq** (recomendado): https://console.groq.com/keys
3. **Mistral**: https://console.mistral.ai/api-keys
4. **Cohere**: https://dashboard.cohere.com/api-keys
5. **HuggingFace**: https://huggingface.co/settings/tokens
6. **Together AI**: https://api.together.xyz/settings/api-keys
7. **OpenRouter**: https://openrouter.ai/keys

Copie `.env.example` para `.env` e preencha as chaves:
```bash
cp .env.example .env
# Edite o .env com suas chaves
```

> **Mínimo:** Configure pelo menos 1 chave. **Ideal:** Configure todas para máxima disponibilidade.

### Passo 2: Processar os livros (rodar UMA VEZ)

```bash
pip install -r requirements_process.txt

# Coloque os PDFs na pasta livros/
mkdir livros
# copie: guyton.pdf, silverthorn.pdf, porto.pdf

python process_books.py
```

### Passo 3: Rodar

```bash
pip install -r requirements.txt
python app.py
```

Acesse: **http://localhost:5000**

---

## 🌐 Deploy Gratuito (Render)

Para acessar de qualquer lugar sem PC ligado:

1. Suba o projeto no **GitHub** (incluindo `data/knowledge_base.json`)
2. Acesse https://render.com → crie conta grátis
3. **New → Web Service** → conecte o repositório
4. Adicione suas chaves de API em **Environment Variables**
5. Deploy automático!

URL: `https://seu-app.onrender.com`

> **Dica:** Use https://uptimerobot.com (grátis) para pingar o serviço a cada 14 min e evitar que ele durma.

---

## 🔧 Como Funciona a Rotação

```
Pergunta do estudante
        │
        ▼
  ┌─ Gemini ──── OK? ──→ Responde ✅
  │     │
  │   Limite!
  │     ▼
  ├─ Groq ────── OK? ──→ Responde ✅
  │     │
  │   Limite!
  │     ▼
  ├─ Mistral ─── OK? ──→ Responde ✅
  │     │
  │   Limite!
  │     ▼
  ├─ Cohere ─── ...continua...
  │
  └─ (todas falharam → aguarde 1 min)
```

O sistema:
- Prioriza APIs de melhor qualidade
- Conta requests por minuto e por dia
- Detecta rate limits (HTTP 429) e faz cooldown automático
- Mostra na interface qual API respondeu
- Painel de status com uso em tempo real

---

## 📁 Estrutura

```
medic-qa/
├── app.py                    # Servidor Flask
├── multi_api.py              # Gerenciador multi-API com rotação
├── process_books.py          # Processador de PDFs (1x)
├── requirements.txt          # Deps servidor
├── requirements_process.txt  # Deps processamento
├── render.yaml               # Config Render
├── .env.example              # Template com TODAS as chaves
├── templates/
│   └── index.html            # Interface do chat
├── data/
│   └── knowledge_base.json   # Base gerada
└── livros/
    └── (seus PDFs)
```

---

## 💡 Dicas

- **Qualidade da resposta** depende do PDF: textos selecionáveis > escaneados
- **Quanto mais APIs configurar**, menor a chance de ficar sem resposta
- **O painel de APIs** (botão 📡) mostra o uso em tempo real de cada provedor
- **Fallback** é indicado na resposta com badge amarelo
- Para PDFs escaneados, use `ocrmypdf` antes para melhorar a extração

---

## 💰 Custo Total

| Componente | Custo |
|-----------|-------|
| 7 APIs de IA | **R$ 0** |
| Hosting (Render) | **R$ 0** |
| Embeddings | **R$ 0** |
| **TOTAL** | **R$ 0/mês** |
