import base64

data = b"Hello, Base64!"
encoded = base64.b64encode(data)
print("Base64 encoded:", encoded)

decoded = base64.b64decode(encoded)
print("Decoded:", decoded)
