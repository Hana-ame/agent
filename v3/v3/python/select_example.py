import select
import socket
import time

# 创建非阻塞 socket
server = socket.socket()
server.setblocking(False)
server.bind(('127.0.0.1', 0))
server.listen(5)
port = server.getsockname()[1]
print(f"Server listening on port {port}")

# 在另一个线程中连接
import threading
def client():
    time.sleep(0.1)
    s = socket.socket()
    s.connect(('127.0.0.1', port))
    s.send(b"Hello select")
    s.close()

threading.Thread(target=client).start()

# select 监控
rlist, wlist, xlist = select.select([server], [], [], 2)
if server in rlist:
    conn, addr = server.accept()
    data = conn.recv(1024)
    print(f"Received: {data.decode()}")
    conn.close()
server.close()
