import threading
import queue
import time

q = queue.Queue()

def producer():
    for i in range(5):
        q.put(i)
        print(f"Produced {i}")
        time.sleep(0.5)
    q.put(None)  # sentinel to stop consumer

def consumer():
    while True:
        item = q.get()
        if item is None:
            break
        print(f"Consumed {item}")
        q.task_done()

t1 = threading.Thread(target=producer)
t2 = threading.Thread(target=consumer)
t1.start()
t2.start()
t1.join()
t2.join()
