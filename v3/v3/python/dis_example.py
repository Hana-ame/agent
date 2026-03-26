import dis

def add(a, b):
    return a + b

print("Bytecode for add:")
dis.dis(add)
