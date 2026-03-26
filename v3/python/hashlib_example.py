import hashlib
import hmac

data = b"hello world"

# MD5
md5 = hashlib.md5(data).hexdigest()
print("MD5:", md5)

# SHA256
sha256 = hashlib.sha256(data).hexdigest()
print("SHA256:", sha256)

# HMAC
key = b"secret"
hmac_obj = hmac.new(key, data, hashlib.sha256)
print("HMAC:", hmac_obj.hexdigest())
