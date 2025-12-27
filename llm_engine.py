import os
import streamlit as st
from typing import Optional

def query_llm(prompt: str) -> str:
    """
    Executes a query against the configured Large Language Model (LLM).
    
    Priority:
    1. Groq Cloud API (Production/High Performance)
    2. Local Ollama Instance (Fallback/Offline Dev)
    
    Args:
        prompt (str): The input prompt for the model.
        
    Returns:
        str: The text response from the model or an error message.
    """
    
    # --- Strategy 1: Groq Cloud Inference ---
    try:
        from groq import Groq
        
        # Retrieve API Key from Secrets or Environment Variables
        api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
        
        if api_key:
            client = Groq(api_key=api_key)
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1024
            )
            return completion.choices[0].message.content
            
    except ImportError:
        pass  # Groq library not present; proceed to fallback.
    except Exception as e:
        print(f"[LLM-Cloud Error]: {e}")

    # --- Strategy 2: Local Ollama Inference ---
    try:
        import ollama
        response = ollama.chat(
            model="llama3.2", 
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response['message']['content']
    except ImportError:
        return "System Error: 'ollama' module not found."
    except Exception as e:
        return f"System Offline: Unable to connect to inference engine. Details: {e}"