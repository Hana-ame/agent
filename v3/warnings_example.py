import warnings

def deprecated_function():
    warnings.warn("This function is deprecated", DeprecationWarning, stacklevel=2)
    return "still works"

with warnings.catch_warnings(record=True) as w:
    result = deprecated_function()
    print("Result:", result)
    print("Warning:", w[0].message)
