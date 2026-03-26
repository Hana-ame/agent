import weakref

class Node:
    def __init__(self, value):
        self.value = value

node = Node(42)
weak_node = weakref.ref(node)
print("Weak reference:", weak_node())
del node
print("After deletion:", weak_node())
