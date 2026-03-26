from datetime import datetime, timedelta

now = datetime.now()
print("Now:", now)
print("Year:", now.year, "Month:", now.month)

# 时间差
future = now + timedelta(days=7, hours=2)
print("One week + 2h later:", future)

# 格式化
formatted = now.strftime("%Y-%m-%d %H:%M:%S")
print("Formatted:", formatted)

# 解析
parsed = datetime.strptime("2024-01-15 12:30:45", "%Y-%m-%d %H:%M:%S")
print("Parsed:", parsed)
