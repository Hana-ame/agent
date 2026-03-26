try:
    import pandas as pd
except ImportError:
    print("pandas not installed, skipping")
    exit(0)

data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'Paris', 'London']
}
df = pd.DataFrame(data)
print("DataFrame:")
print(df)
print("\nDescribe:")
print(df.describe())
print("\nFilter Age > 28:")
print(df[df['Age'] > 28])
