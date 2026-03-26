import secrets

# 安全随机整数
print("Random int (0-100):", secrets.randbelow(101))

# 安全随机字节
print("Random bytes (16):", secrets.token_bytes(16).hex())

# 安全 URL 安全字符串
print("URL-safe token:", secrets.token_urlsafe(16))

# 安全十六进制
print("Hex token:", secrets.token_hex(16))
