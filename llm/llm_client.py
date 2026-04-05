import os
import requests


def _get_groq_key():
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
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 350,
            "temperature": 0.2,
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def ollama_client(prompt, model="llama3.2:3b"):
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False,
              "options": {"num_predict": 300, "temperature": 0.2}},
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
    """Returns (callable, source_name). Groq → Ollama → None."""
    if _get_groq_key():
        return groq_client, "groq"
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        if r.status_code == 200:
            return ollama_client, "ollama"
    except Exception:
        pass
    return None, "none"


def build_safe_prompt(question, evidence_list):
    """
    Build a privacy-safe LLM prompt using ONLY structured metadata.
    Raw contract text is NEVER included.
    """
    meta_blocks = []
    for e in evidence_list[:4]:
        block = f"CLAUSE TYPE: {e['clause']}\nDEVIATING: {'Yes' if e['deviating'] else 'No'}"
        if e["deviating"] and e.get("reasons"):
            block += f"\nFLAGS: {'; '.join(e['reasons'])}"
        if e.get("explanations"):
            block += f"\nCONTEXT: {' '.join(e['explanations'][:2])}"
        if e.get("severity"):
            block += f"\nSEVERITY: {e['severity']}"
        meta_blocks.append(block)

    meta_str = "\n\n".join(meta_blocks) if meta_blocks else "No relevant clauses found."

    return (
        "You are a contract analyst. Using ONLY the structured clause metadata below "
        "(no raw contract text is provided for privacy), answer the user's question "
        "clearly and concisely.\n"
        "Use bullet points. Max 200 words. End with: 'Note: This is not legal advice.'\n\n"
        f"CLAUSE METADATA:\n{meta_str}\n\n"
        f"QUESTION: {question}\n\nANSWER:"
    )
