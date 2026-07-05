"""
Verify Groq API key works with LangChain.
Run: python scripts/verify_groq_api.py
"""
import os
import sys

def verify_groq_api(api_key: str) -> bool:
    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage

        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=api_key,
            max_tokens=50,
        )
        response = llm.invoke([HumanMessage(content="Reply with exactly: GROQ_API_KEY_VERIFIED")])
        content = response.content.strip()
        print(f"[OK] Groq API Key VERIFIED")
        print(f"   Model: llama-3.3-70b-versatile")
        print(f"   Response: {content}")
        return True
    except Exception as e:
        print(f"[FAIL] Groq API Key FAILED: {e}")
        return False


if __name__ == "__main__":
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print("[FAIL] GROQ_API_KEY not set in environment. Add it to your .env file.")
        sys.exit(1)
    success = verify_groq_api(api_key)
    sys.exit(0 if success else 1)
