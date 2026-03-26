import threading
import time

barrier = threading.Barrier(3)

def worker(name):
    print(f"{name} waiting at barrier")
    barrier.wait()
    print(f"{name} passed barrier")

threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
for t in threads:
    t.start()
for t in threads:
    t.join()
