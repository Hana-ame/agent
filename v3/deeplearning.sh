# 创建深度学习示例目录
cd v3/deep_learning

# 1. PyTorch 简单神经网络
cat > pytorch_nn.py << 'EOF'
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    import numpy as np
except ImportError:
    print("PyTorch or scikit-learn not installed, skipping")
    exit(0)

# 生成分类数据
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为 Tensor
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

# 定义简单网络
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleNN(20, 64, 2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练
epochs = 20
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 测试
with torch.no_grad():
    outputs = model(X_test_t)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test_t).float().mean()
    print(f"Test Accuracy: {accuracy.item():.4f}")
EOF

# 2. PyTorch 训练可视化（保存损失曲线）
cat > pytorch_visualize.py << 'EOF'
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("PyTorch, sklearn, or matplotlib not installed, skipping")
    exit(0)

X, y = make_classification(n_samples=500, n_features=10, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 2)
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

losses = []
epochs = 50
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.savefig('loss_curve.png')
print("Loss curve saved as loss_curve.png")
import os
os.remove('loss_curve.png')
print("Removed loss_curve.png")
EOF

# 3. TensorFlow/Keras 简单模型（如果安装）
cat > tensorflow_nn.py << 'EOF'
try:
    import tensorflow as tf
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
except ImportError:
    print("TensorFlow not installed, skipping")
    exit(0)

X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(20,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {acc:.4f}")
EOF

# 4. 使用 PyTorch 的 DataLoader
cat > pytorch_dataloader.py << 'EOF'
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
except ImportError:
    print("PyTorch not installed, skipping")
    exit(0)

class RandomDataset(Dataset):
    def __init__(self, size=1000):
        self.data = torch.randn(size, 10)
        self.labels = torch.randint(0, 2, (size,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

dataset = RandomDataset(500)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

print(f"Dataset size: {len(dataset)}")
for batch_idx, (data, labels) in enumerate(dataloader):
    print(f"Batch {batch_idx}: data shape {data.shape}, labels shape {labels.shape}")
    if batch_idx == 2:
        break
EOF

# 5. 使用 PyTorch 的 GPU 检测（如果有）
cat > gpu_check.py << 'EOF'
try:
    import torch
    if torch.cuda.is_available():
        print(f"CUDA available, device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
    else:
        print("CUDA not available, running on CPU")
except ImportError:
    print("PyTorch not installed, skipping")
EOF

# 运行所有脚本（依赖缺失时跳过）
for py in pytorch_nn.py pytorch_visualize.py tensorflow_nn.py pytorch_dataloader.py gpu_check.py; do
    python "$py" > "${py%.py}.out" 2>&1
    echo "Executed $py"
done

cd ../..
git add v3/deep_learning
git commit -m "Add deep learning examples: PyTorch simple NN, visualization, DataLoader, TensorFlow NN, GPU check"
git tag -a "python-exercise-v32" -m "Deep learning frameworks: PyTorch and TensorFlow basics"

echo ""
echo "========================================"
echo "       深度学习框架练习报告"
echo "========================================"
echo "新增练习: PyTorch 神经网络、训练可视化、DataLoader、TensorFlow 模型、GPU 检测"
echo "所有脚本运行完成（依赖缺失时自动跳过）"
echo "Git 提交完成，标签 python-exercise-v32 已添加"
echo ""
echo "★★★★★"