import multiprocessing
import os

def worker(name):
    print(f"Process {name} (PID: {os.getpid()}) working")
    return name * 2

if __name__ == "__main__":
    with multiprocessing.Pool(processes=3) as pool:
        results = pool.map(worker, ['A', 'B', 'C'])
    print("Results:", results)
