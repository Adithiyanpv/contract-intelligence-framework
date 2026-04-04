import os
import requests


def _get_groq_key():
    """Read Groq API key from env or Streamlit secrets."""
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            pass
    return key


def groq_client(prompt, model="llama-3.1-8b-instant"):
    api_key = _get_groq_key()
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")

    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 400,
            "temperature": 0.3,
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def ollama_client(prompt, model="llama3.2:3b"):
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 300, "temperature": 0.3},
        },
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    if "response" in data:
        return data["response"]
    if "message" in data and "content" in data["message"]:
        return data["message"]["content"]
    return str(data)


def get_llm_client():
    """
    Returns (callable, source_name).
    Priority: Groq (key present) → Ollama (reachable) → None
    Does NOT make a test call — just checks availability.
    """
    # 1. Groq — just check if key exists, no test call
    if _get_groq_key():
        return groq_client, "groq"

    # 2. Ollama — check if server is reachable
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        if r.status_code == 200:
            return ollama_client, "ollama"
    except Exception:
        pass

    return None, "none"
