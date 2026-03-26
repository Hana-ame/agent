from http.server import HTTPServer, BaseHTTPRequestHandler
import json

class JSONHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/api/data':
            length = int(self.headers['Content-Length'])
            data = self.rfile.read(length)
            try:
                payload = json.loads(data)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                response = {'status': 'ok', 'received': payload}
                self.wfile.write(json.dumps(response).encode())
            except json.JSONDecodeError:
                self.send_error(400, 'Invalid JSON')
        else:
            self.send_error(404)

with HTTPServer(('127.0.0.1', 0), JSONHandler) as server:
    port = server.server_address[1]
    print(f"Server on port {port}")
    import threading, urllib.request
    def client():
        data = json.dumps({'name': 'Alice'}).encode()
        req = urllib.request.Request(f'http://127.0.0.1:{port}/api/data',
                                     data=data, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req) as resp:
            print("Response:", resp.read().decode())
    threading.Thread(target=client).start()
    server.handle_request()
