import pickle
import tempfile

data = {'name': 'Alice', 'age': 30, 'scores': [85, 92, 78]}

with tempfile.NamedTemporaryFile() as tmp:
    # 序列化
    pickle.dump(data, tmp)
    tmp.seek(0)
    # 反序列化
    loaded = pickle.load(tmp)
    print("Loaded:", loaded)
    assert data == loaded
