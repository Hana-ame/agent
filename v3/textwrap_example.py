import textwrap

long_text = "The quick brown fox jumps over the lazy dog. " * 3
print("Original length:", len(long_text))

wrapped = textwrap.fill(long_text, width=50)
print("Wrapped:\n", wrapped)

shortened = textwrap.shorten(long_text, width=60, placeholder="...")
print("Shortened:", shortened)

dedented = textwrap.dedent("""
    This is
        indented
    text
""")
print("Dedented:\n", dedented)
