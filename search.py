import requests
from ddgs import DDGS


def search(query: str, max_results=100):
    with DDGS() as ddgs:
        results = ddgs.text(query=query, max_results=max_results)
        return results

def get(url: str):
    r = requests.get("https://en.wikipedia.org/wiki/2010_Nobel_Prize_in_Physics", headers={"User-Agent": "simple ai/0.1"})
    # print(r)
    return r.text    
        


