import csv
import tempfile
import os

data = [
    ['Name', 'Age', 'City'],
    ['Alice', 30, 'New York'],
    ['Bob', 25, 'Boston'],
    ['Charlie', 35, 'Chicago']
]

with tempfile.NamedTemporaryFile(mode='w+', suffix='.csv', newline='') as tmp:
    writer = csv.writer(tmp)
    writer.writerows(data)
    tmp.seek(0)
    reader = csv.reader(tmp)
    for row in reader:
        print(row)
