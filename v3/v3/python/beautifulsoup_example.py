try:
    from bs4 import BeautifulSoup
    import requests
except ImportError:
    print("beautifulsoup4 or requests not installed, skipping")
    exit(0)

html = '<html><body><h1>Hello</h1><p>World</p></body></html>'
soup = BeautifulSoup(html, 'html.parser')
print("Title:", soup.h1.text)
print("Paragraph:", soup.p.text)
