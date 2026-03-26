import uuid

print("UUID4:", uuid.uuid4())
print("UUID1:", uuid.uuid1())
print("UUID3 (namespace DNS):", uuid.uuid3(uuid.NAMESPACE_DNS, "example.com"))
print("UUID5 (namespace URL):", uuid.uuid5(uuid.NAMESPACE_URL, "https://example.com"))
