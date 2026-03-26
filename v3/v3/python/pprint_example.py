import pprint

data = {
    'name': 'Alice',
    'age': 30,
    'hobbies': ['reading', 'coding', 'hiking'],
    'address': {
        'city': 'New York',
        'zip': '10001'
    }
}

print("Default print:")
print(data)

print("\nPretty print:")
pprint.pprint(data, indent=2, width=40, sort_dicts=False)
