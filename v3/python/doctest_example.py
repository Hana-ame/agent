def add(a, b):
    """
    Return the sum of a and b.

    >>> add(2, 3)
    5
    >>> add(-1, 1)
    0
    """
    return a + b

def multiply(a, b):
    """
    Return the product of a and b.

    >>> multiply(2, 3)
    6
    >>> multiply(-2, 3)
    -6
    """
    return a * b

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
