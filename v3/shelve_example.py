import shelve
import tempfile
import os

with tempfile.TemporaryDirectory() as tmpdir:
    db_path = os.path.join(tmpdir, 'mydb')
    with shelve.open(db_path) as db:
        db['key1'] = 'value1'
        db['key2'] = [1, 2, 3]
        db['key3'] = {'a': 1}
    
    with shelve.open(db_path) as db:
        print("Keys:", list(db.keys()))
        print("key1:", db['key1'])
        print("key2:", db['key2'])
        print("key3:", db['key3'])
