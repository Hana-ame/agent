try:
    from sklearn.cluster import KMeans
    import numpy as np
except ImportError:
    print("scikit-learn not installed, skipping")
    exit(0)

np.random.seed(42)
X = np.random.rand(100, 2) * 10
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_
centers = kmeans.cluster_centers_

print("Cluster centers:\n", centers)
print("First 10 labels:", labels[:10])
