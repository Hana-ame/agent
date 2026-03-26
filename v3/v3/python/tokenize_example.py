import tokenize
import io

code = b"def foo():\n    return 42"
for token in tokenize.tokenize(io.BytesIO(code).readline):
    print(f"{tokenize.tok_name[token.type]:10} {token.string!r}")
