import os
from groq import Groq
import streamlit as st

# --- CONFIGURATION ---
# Primary: High Intelligence
PRIMARY_MODEL = "llama-3.3-70b-versatile"

# Fallback: The NEW supported model
FALLBACK_MODEL = "llama-3.1-8b-instant" 

# --- AUTHENTICATION ---
try:
    api_key = st.secrets["GROQ_API_KEY"]
except:
    try:
        api_key = os.environ.get("GROQ_API_KEY")
    except:
        api_key = None

# Initialize Client
if api_key:
    client = Groq(api_key=api_key)
else:
    client = None

def query_llm(prompt):
    if not client:
        return "⚠️ AI Error: Groq API Key not found. Add GROQ_API_KEY to Streamlit Secrets."

    try:
        # Attempt 1: Primary Model
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=PRIMARY_MODEL,
            temperature=0.3,
        )
        return chat_completion.choices[0].message.content

    except Exception as e_primary:
        # Attempt 2: Fallback Model
        error_msg = str(e_primary).lower()
        
        # Catch Rate Limits (429) OR Decommissioned errors
        if "429" in error_msg or "rate limit" in error_msg or "error" in error_msg:
            try:
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=FALLBACK_MODEL,
                    temperature=0.3,
                )
                return f"{chat_completion.choices[0].message.content}\n\n*(Backup Model used)*"
            except Exception as e_fallback:
                return f"⚠️ System Busy: {str(e_fallback)}"
        else:
            return f"⚠️ AI Error: {str(e_primary)}"