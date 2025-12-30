import os
from groq import Groq
import streamlit as st

# --- CONFIGURATION ---
# Primary: High Intelligence (Best for detailed reports)
PRIMARY_MODEL = "llama-3.3-70b-versatile"

# Fallback: High Speed/Efficiency (Best for avoiding limits)
FALLBACK_MODEL = "llama3-8b-8192" 

# --- AUTHENTICATION ---
try:
    # Option 1: Cloud Secrets
    api_key = st.secrets["GROQ_API_KEY"]
except:
    try:
        # Option 2: Local Env
        api_key = os.environ.get("GROQ_API_KEY")
    except:
        api_key = None

# Initialize Client
if api_key:
    client = Groq(api_key=api_key)
else:
    client = None

def query_llm(prompt):
    """
    Robust Query Function with Fallback Logic.
    1. Tries PRIMARY_MODEL (70b).
    2. If it fails (Rate Limit/Error), switches to FALLBACK_MODEL (8b).
    """
    if not client:
        return "⚠️ AI Error: Groq API Key not found."

    # --- ATTEMPT 1: PRIMARY MODEL ---
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=PRIMARY_MODEL,
            temperature=0.3,
        )
        return chat_completion.choices[0].message.content

    except Exception as e_primary:
        # --- ATTEMPT 2: FALLBACK MODEL ---
        # Check if error is related to Rate Limit (429) or other API issues
        error_msg = str(e_primary).lower()
        
        if "429" in error_msg or "rate limit" in error_msg:
            try:
                # Retry with smaller model
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=FALLBACK_MODEL,
                    temperature=0.3,
                )
                # Success! Return result but maybe log that we used fallback (optional)
                return f"{chat_completion.choices[0].message.content}\n\n*(Generated via Backup Model due to high traffic)*"
            
            except Exception as e_fallback:
                return f"⚠️ System Overload: Both primary and backup AI models are currently busy. Please try again in 60 seconds.\nError: {str(e_fallback)}"
        
        else:
            # If it's not a rate limit (e.g., connection error), return the error
            return f"⚠️ AI Error: {str(e_primary)}"