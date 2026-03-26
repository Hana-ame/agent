from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import time

class SimpleHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Hello from simple server!")

server = HTTPServer(('localhost', 0), SimpleHandler)
port = server.server_address[1]
print(f"Serving on port {port}")

def run_server():
    server.serve_forever()

thread = threading.Thread(target=run_server, daemon=True)
thread.start()
time.sleep(0.5)

# 测试请求
import urllib.request
with urllib.request.urlopen(f'http://localhost:{port}/') as response:
    print(response.read().decode())

server.shutdown()
server.server_close()
