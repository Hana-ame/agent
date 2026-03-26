from urllib.parse import urlparse, urlunparse, urlencode, parse_qs

url = "https://example.com:8080/path/to/page?name=Alice&age=30#section"
parsed = urlparse(url)
print("Scheme:", parsed.scheme)
print("Netloc:", parsed.netloc)
print("Path:", parsed.path)
print("Query:", parsed.query)
print("Fragment:", parsed.fragment)

# 构建 URL
components = ('https', 'example.com', '/path', '', 'name=Bob', 'section')
new_url = urlunparse(components)
print("New URL:", new_url)

# 查询参数编码
params = {'name': 'Alice', 'age': 30, 'city': 'New York'}
encoded = urlencode(params)
print("Encoded params:", encoded)

# 解析查询字符串
query_str = "name=Alice&age=30&city=New+York"
parsed_qs = parse_qs(query_str)
print("Parsed query string:", parsed_qs)
