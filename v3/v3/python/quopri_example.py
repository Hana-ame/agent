import quopri

original = b"Hello = World & more"
encoded = quopri.encodestring(original)
print(f"Original: {original}")
print(f"Encoded: {encoded}")
decoded = quopri.decodestring(encoded)
print(f"Decoded: {decoded}")
