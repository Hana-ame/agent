import threading
import time
import random

sem = threading.Semaphore(3)  # 允许3个线程同时执行

def worker(name):
    sem.acquire()
    print(f"{name} acquired semaphore")
    time.sleep(random.uniform(0.5, 1.5))
    print(f"{name} releasing")
    sem.release()

threads = []
for i in range(10):
    t = threading.Thread(target=worker, args=(f"Thread-{i}",))
    threads.append(t)
    t.start()

for t in threads:
    t.join()
print("All done")
