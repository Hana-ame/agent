try:
    import jsonschema
    from jsonschema import validate
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False
    print("jsonschema not installed, skipping demo")
    exit(0)

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "number", "minimum": 0}
    },
    "required": ["name"]
}

data_valid = {"name": "Alice", "age": 30}
data_invalid = {"name": "Bob", "age": -5}

validate(instance=data_valid, schema=schema)
print("Valid data passes")

try:
    validate(instance=data_invalid, schema=schema)
except jsonschema.ValidationError as e:
    print("Invalid data:", e.message)
