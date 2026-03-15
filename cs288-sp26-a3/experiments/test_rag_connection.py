import json
import os
import sys

# Ensure current dir is in path
sys.path.append(os.getcwd())

from experiments.evaluate_rag import run_evaluation

def test_small_rag():
    # Verify API key
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        print("\n[ERROR] OPENROUTER_API_KEY is NOT set in this session.")
        print("Please run the following before this script:")
        print("export OPENROUTER_API_KEY='your_key_here'\n")
        return

    print("="*40)
    print("RUNNING SMALL RAG TEST (2 QUESTIONS)")
    print("="*40)
    
    try:
        run_evaluation("data/test_small.jsonl", k=3)
        print("\n[SUCCESS] Pipeline completed. Check results above.")
    except Exception as e:
        print(f"\n[FAILED] Pipeline encountered an error: {e}")

if __name__ == "__main__":
    test_small_rag()
