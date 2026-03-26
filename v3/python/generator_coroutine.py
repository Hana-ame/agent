def subgenerator():
    yield 1
    yield 2
    yield 3

def delegator():
    yield from subgenerator()
    yield 4

if __name__ == "__main__":
    for value in delegator():
        print(value)
