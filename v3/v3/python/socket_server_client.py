import socket
import threading
import time

def server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        s.listen()
        port = s.getsockname()[1]
        print(f"Server listening on port {port}")
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            data = conn.recv(1024)
            print("Received:", data.decode())
            conn.sendall(b"Hello from server")
        return port

def client(port):
    time.sleep(0.1)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(('127.0.0.1', port))
        s.sendall(b"Hello from client")
        data = s.recv(1024)
        print("Client received:", data.decode())

if __name__ == "__main__":
    port = server()
    client(port)
