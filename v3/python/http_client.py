import urllib.request

def fetch(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            data = response.read().decode('utf-8')
            print(f"Status: {response.status}")
            print(f"Content length: {len(data)}")
            print("First 100 chars:", data[:100])
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    fetch("http://example.com")
