import sys

def describe(obj):
    match obj:
        case int() as n:
            return f"Integer: {n}"
        case str() as s:
            return f"String: {s}"
        case list() as lst:
            return f"List with {len(lst)} items"
        case dict() as dct:
            return f"Dict with {len(dct)} keys"
        case _:
            return "Unknown type"

if __name__ == "__main__":
    print(describe(42))
    print(describe("hello"))
    print(describe([1,2,3]))
    print(describe({"a":1}))
    print(describe(None))
