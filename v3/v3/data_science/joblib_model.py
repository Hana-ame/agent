try:
    from sklearn.linear_model import LinearRegression
    from sklearn.datasets import make_regression
    import joblib
    import os
except ImportError:
    print("scikit-learn or joblib not installed, skipping")
    exit(0)

# 生成数据
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
model = LinearRegression()
model.fit(X, y)

# 保存模型
joblib.dump(model, 'model.joblib')
print("Model saved as model.joblib")

# 加载模型
loaded_model = joblib.load('model.joblib')
print("Model loaded")
print("Prediction for X=5:", loaded_model.predict([[5]]))

# 清理
os.remove('model.joblib')
