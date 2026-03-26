try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("pandas or numpy not installed, skipping")
    exit(0)

# 创建示例数据
np.random.seed(42)
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'age': np.random.randint(20, 40, 5),
    'salary': np.random.randint(40000, 80000, 5),
    'department': ['IT', 'HR', 'IT', 'Finance', 'HR']
}
df = pd.DataFrame(data)
print("DataFrame:")
print(df)

print("\nBasic statistics:")
print(df.describe())

print("\nGroup by department:")
print(df.groupby('department')['salary'].mean())

print("\nFilter salary > 50000:")
print(df[df['salary'] > 50000])
