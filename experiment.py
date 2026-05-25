from loop import run_once
import time
from pathlib import Path

def experiment():
    print("Starting experiment: calling run_once()...")
    try:
        result = run_once()
        print("run_once() executed successfully.")
        print(f"Result snippet: {result[:200]}...")
    except Exception as e:
        print(f"run_once() failed: {e}")

if __name__ == "__main__":
    experiment()
