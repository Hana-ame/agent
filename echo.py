import json

# 从文件加载JSON数据
with open('history.json', 'r', encoding='utf-8') as file:
    data_list = json.load(file)

# 直接访问最后一项的content
print(data_list[-1]['content'])