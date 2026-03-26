try:
    from tqdm import tqdm
    import time
except ImportError:
    print("tqdm not installed, skipping")
    exit(0)

for i in tqdm(range(100), desc="Processing"):
    time.sleep(0.01)
print("Done")
