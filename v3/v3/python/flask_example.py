try:
    from flask import Flask, jsonify
except ImportError:
    print("flask not installed, skipping")
    exit(0)

app = Flask(__name__)

@app.route('/api/hello/<name>')
def hello(name):
    return jsonify(message=f"Hello, {name}!")

# 使用测试客户端
client = app.test_client()
resp = client.get('/api/hello/Alice')
print("Response status:", resp.status_code)
print("Response data:", resp.get_json())
