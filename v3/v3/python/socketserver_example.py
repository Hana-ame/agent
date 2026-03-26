import socketserver
import threading

class MyHandler(socketserver.StreamRequestHandler):
    def handle(self):
        data = self.rfile.readline().strip()
        print(f"Received: {data.decode()}")
        self.wfile.write(b"Echo: " + data + b"\n")

with socketserver.TCPServer(('127.0.0.1', 0), MyHandler) as server:
    port = server.server_address[1]
    print(f"Server on port {port}")

    def client():
        import socket
        s = socket.socket()
        s.connect(('127.0.0.1', port))
        s.send(b"Hello server\n")
        response = s.recv(1024)
        print(f"Client received: {response.decode().strip()}")
        s.close()

    threading.Thread(target=client).start()
    server.handle_request()
