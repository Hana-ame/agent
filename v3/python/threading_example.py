import threading
import time

def worker(name, delay):
    print(f"Thread {name} starting")
    time.sleep(delay)
    print(f"Thread {name} finished after {delay}s")

threads = []
for i in range(3):
    t = threading.Thread(target=worker, args=(i, i+1))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print("All threads done")
