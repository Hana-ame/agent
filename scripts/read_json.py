import json

def load_config(filepath='config.json'):
    with open(filepath, 'r') as f:
        return json.load(f)

config = load_config()
print(f"App name: {config.get('name')}")