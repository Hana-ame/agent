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
