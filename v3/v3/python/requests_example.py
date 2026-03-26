try:
    import requests
except ImportError:
    print("requests not installed, skipping")
    exit(0)

resp = requests.get('https://httpbin.org/get', params={'name': 'Alice'})
print(f"Status: {resp.status_code}")
print(f"URL: {resp.url}")
print(f"JSON response: {resp.json()['args']}")
