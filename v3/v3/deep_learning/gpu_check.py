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
