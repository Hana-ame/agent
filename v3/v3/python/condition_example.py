import threading
import time

class Buffer:
    def __init__(self, size):
        self.size = size
        self.buffer = []
        self.cond = threading.Condition()

    def produce(self, item):
        with self.cond:
            while len(self.buffer) >= self.size:
                self.cond.wait()
            self.buffer.append(item)
            print(f"Produced: {item}")
            self.cond.notify()

    def consume(self):
        with self.cond:
            while not self.buffer:
                self.cond.wait()
            item = self.buffer.pop(0)
            print(f"Consumed: {item}")
            self.cond.notify()
            return item

buffer = Buffer(3)

def producer():
    for i in range(5):
        buffer.produce(i)
        time.sleep(0.2)

def consumer():
    for _ in range(5):
        buffer.consume()
        time.sleep(0.3)

threads = [threading.Thread(target=producer), threading.Thread(target=consumer)]
for t in threads:
    t.start()
for t in threads:
    t.join()
