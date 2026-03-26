try:
    import numpy as np
except ImportError:
    print("numpy not installed, skipping")
    exit(0)

arr = np.array([1, 2, 3, 4, 5])
print("Array:", arr)
print("Mean:", np.mean(arr))
print("Std:", np.std(arr))
print("Squared:", arr ** 2)
matrix = np.array([[1,2],[3,4]])
print("Matrix:\n", matrix)
print("Matrix transpose:\n", matrix.T)
