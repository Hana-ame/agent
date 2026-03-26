try:
    from pydantic import BaseModel, ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    print("pydantic not installed, skipping demo")
    exit(0)

class User(BaseModel):
    name: str
    age: int
    email: str

if __name__ == "__main__":
    user = User(name="Alice", age=30, email="alice@example.com")
    print(user)
    try:
        User(name="Bob", age="twenty", email="bob")
    except ValidationError as e:
        print("Validation error:", e)
