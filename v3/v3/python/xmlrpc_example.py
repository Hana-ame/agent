import xmlrpc.server
import xmlrpc.client
import threading

def add(a, b):
    return a + b

with xmlrpc.server.SimpleXMLRPCServer(('127.0.0.1', 0)) as server:
    port = server.server_address[1]
    server.register_function(add, 'add')
    print(f"XML-RPC server on port {port}")

    def client():
        proxy = xmlrpc.client.ServerProxy(f"http://127.0.0.1:{port}")
        result = proxy.add(5, 7)
        print(f"Client result: {result}")
    
    threading.Thread(target=client).start()
    server.handle_request()
