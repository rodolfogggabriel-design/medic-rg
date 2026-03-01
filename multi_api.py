"""
=============================================================
  MedicQA - Gerenciador Multi-API (Free Tier Rotation)
=============================================================
Combina TODAS as APIs gratuitas disponíveis e faz rotação
automática quando uma atinge o limite. Resultado: uso
praticamente ilimitado sem pagar nada.

APIs suportadas:
  1. Google Gemini    — 15 req/min, 1500 req/dia
  2. Groq             — 30 req/min, 14400 req/dia (Llama/Mixtral)
  3. Cohere           — 20 req/min, 1000 req/mês
  4. Mistral          — 1 req/s, sem limite diário claro
  5. HuggingFace      — 1000 req/dia (modelos variados)
  6. Together AI      — $5 crédito grátis no signup
  7. OpenRouter       — free models disponíveis
"""

import os
import time
import json
import requests
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class APIProvider:
    """Representa um provedor de API."""
    name: str
    api_key: str
    model: str
    rpm_limit: int          # Requests per minute
    daily_limit: int        # Requests per day (0 = sem limite)
    priority: int           # Menor = maior prioridade
    enabled: bool = True
    requests_this_minute: int = 0
    requests_today: int = 0
    last_minute_reset: float = 0
    last_day_reset: float = 0
    consecutive_errors: int = 0
    cooldown_until: float = 0


class MultiAPIManager:
    """
    Gerencia múltiplas APIs gratuitas com rotação automática.
    Quando uma API atinge o limite, usa a próxima disponível.
    """

    def __init__(self):
        self.providers: list[APIProvider] = []
        self.stats = {
            "total_requests": 0,
            "requests_per_provider": {},
            "errors_per_provider": {},
            "fallbacks": 0
        }
        self._load_providers()

    def _load_providers(self):
        """Carrega todos os provedores configurados."""

        # 1. GOOGLE GEMINI — Melhor qualidade no free tier
        key = os.getenv("GEMINI_API_KEY", "")
        if key and key != "cole_sua_chave_aqui":
            self.providers.append(APIProvider(
                name="Gemini",
                api_key=key,
                model="gemini-2.0-flash",
                rpm_limit=15,
                daily_limit=1500,
                priority=1
            ))

        # 2. GROQ — Mais rápido, limites generosos
        key = os.getenv("GROQ_API_KEY", "")
        if key and key != "cole_sua_chave_aqui":
            self.providers.append(APIProvider(
                name="Groq",
                api_key=key,
                model="llama-3.1-8b-instant",
                rpm_limit=30,
                daily_limit=14400,
                priority=2
            ))

        # 3. MISTRAL — Boa qualidade, free tier
        key = os.getenv("MISTRAL_API_KEY", "")
        if key and key != "cole_sua_chave_aqui":
            self.providers.append(APIProvider(
                name="Mistral",
                api_key=key,
                model="mistral-small-latest",
                rpm_limit=60,
                daily_limit=0,  # Baseado em tokens, não requests
                priority=3
            ))

        # 4. COHERE — Tem RAG nativo, bom para Q&A
        key = os.getenv("COHERE_API_KEY", "")
        if key and key != "cole_sua_chave_aqui":
            self.providers.append(APIProvider(
                name="Cohere",
                api_key=key,
                model="command-r",
                rpm_limit=20,
                daily_limit=1000,
                priority=4
            ))

        # 5. HUGGINGFACE — Fallback com modelos open-source
        key = os.getenv("HUGGINGFACE_API_KEY", "")
        if key and key != "cole_sua_chave_aqui":
            self.providers.append(APIProvider(
                name="HuggingFace",
                api_key=key,
                model="mistralai/Mistral-7B-Instruct-v0.3",
                rpm_limit=10,
                daily_limit=1000,
                priority=5
            ))

        # 6. TOGETHER AI — $5 crédito grátis
        key = os.getenv("TOGETHER_API_KEY", "")
        if key and key != "cole_sua_chave_aqui":
            self.providers.append(APIProvider(
                name="Together",
                api_key=key,
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                rpm_limit=60,
                daily_limit=0,
                priority=6
            ))

        # 7. OPENROUTER — Modelos gratuitos disponíveis
        key = os.getenv("OPENROUTER_API_KEY", "")
        if key and key != "cole_sua_chave_aqui":
            self.providers.append(APIProvider(
                name="OpenRouter",
                api_key=key,
                model="meta-llama/llama-3.3-8b-instruct:free",
                rpm_limit=20,
                daily_limit=0,
                priority=7
            ))

        # Ordena por prioridade
        self.providers.sort(key=lambda p: p.priority)

        # Inicializa stats
        for p in self.providers:
            self.stats["requests_per_provider"][p.name] = 0
            self.stats["errors_per_provider"][p.name] = 0

        print(f"📡 {len(self.providers)} API(s) configurada(s):")
        for p in self.providers:
            print(f"   {'✅' if p.enabled else '❌'} {p.name} ({p.model}) — {p.rpm_limit} req/min")

        if not self.providers:
            print("⚠️  Nenhuma API configurada! Adicione pelo menos uma chave no .env")

    def _reset_counters(self, provider: APIProvider):
        """Reseta contadores de rate limit."""
        now = time.time()

        # Reset por minuto
        if now - provider.last_minute_reset >= 60:
            provider.requests_this_minute = 0
            provider.last_minute_reset = now

        # Reset diário
        if now - provider.last_day_reset >= 86400:
            provider.requests_today = 0
            provider.last_day_reset = now

    def _is_available(self, provider: APIProvider) -> bool:
        """Verifica se o provedor está disponível."""
        if not provider.enabled:
            return False

        now = time.time()

        # Em cooldown por erros consecutivos
        if now < provider.cooldown_until:
            return False

        self._reset_counters(provider)

        # Verifica limites
        if provider.requests_this_minute >= provider.rpm_limit:
            return False

        if provider.daily_limit > 0 and provider.requests_today >= provider.daily_limit:
            return False

        return True

    def _get_next_provider(self) -> Optional[APIProvider]:
        """Retorna o próximo provedor disponível."""
        for provider in self.providers:
            if self._is_available(provider):
                return provider
        return None

    def _call_gemini(self, provider: APIProvider, system_prompt: str, messages: list) -> str:
        """Chama a API do Google Gemini."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{provider.model}:generateContent?key={provider.api_key}"

        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })

        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": contents,
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 2048}
        }

        resp = requests.post(url, json=payload, timeout=30)
        if resp.status_code == 200:
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        elif resp.status_code == 429:
            raise RateLimitError(f"Gemini rate limit: {resp.text}")
        else:
            raise APIError(f"Gemini error {resp.status_code}: {resp.text}")

    def _call_groq(self, provider: APIProvider, system_prompt: str, messages: list) -> str:
        """Chama a API da Groq."""
        url = "https://api.groq.com/openai/v1/chat/completions"

        msgs = [{"role": "system", "content": system_prompt}]
        for msg in messages:
            msgs.append({"role": "assistant" if msg["role"] == "model" else msg["role"], "content": msg["content"]})

        payload = {
            "model": provider.model,
            "messages": msgs,
            "temperature": 0.3,
            "max_tokens": 2048
        }

        resp = requests.post(url, json=payload, headers={
            "Authorization": f"Bearer {provider.api_key}",
            "Content-Type": "application/json"
        }, timeout=30)

        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        elif resp.status_code == 429:
            raise RateLimitError(f"Groq rate limit: {resp.text}")
        else:
            raise APIError(f"Groq error {resp.status_code}: {resp.text}")

    def _call_mistral(self, provider: APIProvider, system_prompt: str, messages: list) -> str:
        """Chama a API da Mistral."""
        url = "https://api.mistral.ai/v1/chat/completions"

        msgs = [{"role": "system", "content": system_prompt}]
        for msg in messages:
            role = msg["role"] if msg["role"] != "model" else "assistant"
            msgs.append({"role": role, "content": msg["content"]})

        payload = {
            "model": provider.model,
            "messages": msgs,
            "temperature": 0.3,
            "max_tokens": 2048
        }

        resp = requests.post(url, json=payload, headers={
            "Authorization": f"Bearer {provider.api_key}",
            "Content-Type": "application/json"
        }, timeout=30)

        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        elif resp.status_code == 429:
            raise RateLimitError(f"Mistral rate limit: {resp.text}")
        else:
            raise APIError(f"Mistral error {resp.status_code}: {resp.text}")

    def _call_cohere(self, provider: APIProvider, system_prompt: str, messages: list) -> str:
        """Chama a API da Cohere."""
        url = "https://api.cohere.ai/v2/chat"

        msgs = [{"role": "system", "content": system_prompt}]
        for msg in messages:
            role = msg["role"] if msg["role"] != "model" else "assistant"
            msgs.append({"role": role, "content": msg["content"]})

        payload = {
            "model": provider.model,
            "messages": msgs,
            "temperature": 0.3,
            "max_tokens": 2048
        }

        resp = requests.post(url, json=payload, headers={
            "Authorization": f"Bearer {provider.api_key}",
            "Content-Type": "application/json"
        }, timeout=30)

        if resp.status_code == 200:
            data = resp.json()
            return data["message"]["content"][0]["text"]
        elif resp.status_code == 429:
            raise RateLimitError(f"Cohere rate limit: {resp.text}")
        else:
            raise APIError(f"Cohere error {resp.status_code}: {resp.text}")

    def _call_huggingface(self, provider: APIProvider, system_prompt: str, messages: list) -> str:
        """Chama a API da HuggingFace Inference."""
        url = f"https://api-inference.huggingface.co/models/{provider.model}/v1/chat/completions"

        msgs = [{"role": "system", "content": system_prompt}]
        for msg in messages:
            role = msg["role"] if msg["role"] != "model" else "assistant"
            msgs.append({"role": role, "content": msg["content"]})

        payload = {
            "messages": msgs,
            "temperature": 0.3,
            "max_tokens": 2048
        }

        resp = requests.post(url, json=payload, headers={
            "Authorization": f"Bearer {provider.api_key}",
            "Content-Type": "application/json"
        }, timeout=60)

        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        elif resp.status_code == 429:
            raise RateLimitError(f"HuggingFace rate limit: {resp.text}")
        else:
            raise APIError(f"HuggingFace error {resp.status_code}: {resp.text}")

    def _call_together(self, provider: APIProvider, system_prompt: str, messages: list) -> str:
        """Chama a API da Together AI."""
        url = "https://api.together.xyz/v1/chat/completions"

        msgs = [{"role": "system", "content": system_prompt}]
        for msg in messages:
            role = msg["role"] if msg["role"] != "model" else "assistant"
            msgs.append({"role": role, "content": msg["content"]})

        payload = {
            "model": provider.model,
            "messages": msgs,
            "temperature": 0.3,
            "max_tokens": 2048
        }

        resp = requests.post(url, json=payload, headers={
            "Authorization": f"Bearer {provider.api_key}",
            "Content-Type": "application/json"
        }, timeout=30)

        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        elif resp.status_code == 429:
            raise RateLimitError(f"Together rate limit: {resp.text}")
        else:
            raise APIError(f"Together error {resp.status_code}: {resp.text}")

    def _call_openrouter(self, provider: APIProvider, system_prompt: str, messages: list) -> str:
        """Chama a API do OpenRouter."""
        url = "https://openrouter.ai/api/v1/chat/completions"

        msgs = [{"role": "system", "content": system_prompt}]
        for msg in messages:
            role = msg["role"] if msg["role"] != "model" else "assistant"
            msgs.append({"role": role, "content": msg["content"]})

        payload = {
            "model": provider.model,
            "messages": msgs,
            "temperature": 0.3,
            "max_tokens": 2048
        }

        resp = requests.post(url, json=payload, headers={
            "Authorization": f"Bearer {provider.api_key}",
            "Content-Type": "application/json"
        }, timeout=30)

        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        elif resp.status_code == 429:
            raise RateLimitError(f"OpenRouter rate limit: {resp.text}")
        else:
            raise APIError(f"OpenRouter error {resp.status_code}: {resp.text}")

    def _call_provider(self, provider: APIProvider, system_prompt: str, messages: list) -> str:
        """Roteia a chamada para o provedor correto."""
        dispatch = {
            "Gemini": self._call_gemini,
            "Groq": self._call_groq,
            "Mistral": self._call_mistral,
            "Cohere": self._call_cohere,
            "HuggingFace": self._call_huggingface,
            "Together": self._call_together,
            "OpenRouter": self._call_openrouter,
        }

        func = dispatch.get(provider.name)
        if not func:
            raise APIError(f"Provedor desconhecido: {provider.name}")

        return func(provider, system_prompt, messages)

    def generate(self, system_prompt: str, messages: list) -> dict:
        """
        Gera uma resposta usando a próxima API disponível.
        Faz fallback automático se uma API falhar.

        Retorna:
            {
                "text": "resposta do modelo",
                "provider": "nome do provedor usado",
                "fallback": True/False se usou fallback
            }
        """
        if not self.providers:
            return {
                "text": "❌ Nenhuma API configurada. Adicione pelo menos uma chave no arquivo .env",
                "provider": "none",
                "fallback": False
            }

        tried = []
        fallback = False

        for provider in self.providers:
            if not self._is_available(provider):
                continue

            if tried:
                fallback = True

            tried.append(provider.name)

            try:
                # Registra a tentativa
                provider.requests_this_minute += 1
                provider.requests_today += 1

                # Faz a chamada
                result = self._call_provider(provider, system_prompt, messages)

                # Sucesso! Reseta erros
                provider.consecutive_errors = 0
                self.stats["total_requests"] += 1
                self.stats["requests_per_provider"][provider.name] += 1
                if fallback:
                    self.stats["fallbacks"] += 1

                return {
                    "text": result,
                    "provider": provider.name,
                    "fallback": fallback
                }

            except RateLimitError as e:
                print(f"⚠️  Rate limit em {provider.name}: {e}")
                provider.consecutive_errors += 1
                # Cooldown proporcional aos erros
                provider.cooldown_until = time.time() + (60 * provider.consecutive_errors)
                self.stats["errors_per_provider"][provider.name] += 1
                continue

            except (APIError, requests.exceptions.Timeout, Exception) as e:
                print(f"❌ Erro em {provider.name}: {e}")
                provider.consecutive_errors += 1
                if provider.consecutive_errors >= 5:
                    provider.cooldown_until = time.time() + 300  # 5 min cooldown
                self.stats["errors_per_provider"][provider.name] += 1
                continue

        # Nenhum provedor disponível
        return {
            "text": f"⏳ Todas as APIs atingiram o limite. Tentei: {', '.join(tried) if tried else 'nenhuma disponível'}. Aguarde ~1 minuto.",
            "provider": "none",
            "fallback": True
        }

    def get_status(self) -> dict:
        """Retorna status de todos os provedores."""
        status = []
        for p in self.providers:
            self._reset_counters(p)
            status.append({
                "name": p.name,
                "model": p.model,
                "enabled": p.enabled,
                "available": self._is_available(p),
                "rpm_used": f"{p.requests_this_minute}/{p.rpm_limit}",
                "daily_used": f"{p.requests_today}/{p.daily_limit}" if p.daily_limit > 0 else f"{p.requests_today}/∞",
                "errors": p.consecutive_errors,
                "priority": p.priority
            })
        return {
            "providers": status,
            "stats": self.stats
        }


class RateLimitError(Exception):
    pass


class APIError(Exception):
    pass
