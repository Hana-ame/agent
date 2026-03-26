import threading
import time

event = threading.Event()

def waiter():
    print("Waiter waiting...")
    event.wait()
    print("Waiter got event!")

def setter():
    print("Setter sleeping...")
    time.sleep(2)
    print("Setter setting event")
    event.set()

threads = [threading.Thread(target=waiter), threading.Thread(target=setter)]
for t in threads:
    t.start()
for t in threads:
    t.join()
