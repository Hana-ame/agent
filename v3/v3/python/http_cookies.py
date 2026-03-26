from http.cookies import SimpleCookie

cookie = SimpleCookie()
cookie["session"] = "abc123"
cookie["session"]["max-age"] = 3600
cookie["user"] = "Alice"
cookie["user"]["path"] = "/"

print("Cookie output:")
print(cookie)

# 解析 Cookie 头
header = "session=abc123; user=Alice"
parsed = SimpleCookie(header)
print("Parsed:")
for key, morsel in parsed.items():
    print(f"{key} = {morsel.value} (attributes: {dict(morsel)})")
