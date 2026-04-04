import os
import requests


def groq_client(prompt, model="llama-3.1-8b-instant"):
    """
    Groq cloud LLM — fast, free tier available.
    Requires GROQ_API_KEY in environment or Streamlit secrets.
    """
    api_key = os.environ.get("GROQ_API_KEY", "")

    # Also try Streamlit secrets if running in cloud
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            pass

    if not api_key:
        return None  # signal: no key available

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
    """
    Local Ollama LLM — used when running locally.
    """
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": 300,
                "temperature": 0.3,
                "top_p": 0.9,
            },
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
    Returns the best available LLM client:
    1. Groq (cloud, fast) — if GROQ_API_KEY is set
    2. Ollama (local) — if server is reachable
    3. None — deterministic fallback
    """
    # Try Groq first
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            pass

    if api_key:
        # Validate key works
        try:
            test = groq_client("Say OK", model="llama-3.1-8b-instant")
            if test:
                return groq_client, "groq"
        except Exception:
            pass

    # Try Ollama
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        if r.status_code == 200:
            return ollama_client, "ollama"
    except Exception:
        pass

    return None, "none"
