try:
    from fastapi import FastAPI, HTTPException, Depends
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("fastapi or uvicorn not installed, skipping demo")
    exit(0)

app = FastAPI()

# 内存存储
todos = []
next_id = 1

class TodoCreate(BaseModel):
    title: str
    description: str = ""

class Todo(TodoCreate):
    id: int
    completed: bool = False

@app.get("/")
def read_root():
    return {"message": "Todo API"}

@app.get("/todos", response_model=list[Todo])
def list_todos():
    return todos

@app.post("/todos", response_model=Todo, status_code=201)
def create_todo(todo: TodoCreate):
    global next_id
    new_todo = Todo(id=next_id, **todo.dict())
    todos.append(new_todo)
    next_id += 1
    return new_todo

@app.get("/todos/{todo_id}", response_model=Todo)
def get_todo(todo_id: int):
    for todo in todos:
        if todo.id == todo_id:
            return todo
    raise HTTPException(status_code=404, detail="Todo not found")

@app.put("/todos/{todo_id}", response_model=Todo)
def update_todo(todo_id: int, todo_update: TodoCreate):
    for todo in todos:
        if todo.id == todo_id:
            todo.title = todo_update.title
            todo.description = todo_update.description
            return todo
    raise HTTPException(status_code=404, detail="Todo not found")

@app.delete("/todos/{todo_id}", status_code=204)
def delete_todo(todo_id: int):
    global todos
    for i, todo in enumerate(todos):
        if todo.id == todo_id:
            todos.pop(i)
            return
    raise HTTPException(status_code=404, detail="Todo not found")

# 使用测试客户端进行简单测试（不启动服务器）
from fastapi.testclient import TestClient
client = TestClient(app)

print("Testing API endpoints:")
response = client.get("/")
print(f"GET /: {response.json()}")

response = client.post("/todos", json={"title": "Learn FastAPI", "description": "Study FastAPI"})
todo = response.json()
print(f"POST /todos: {todo}")

response = client.get("/todos")
print(f"GET /todos: {response.json()}")

response = client.put(f"/todos/{todo['id']}", json={"title": "Master FastAPI", "description": "Deep dive"})
print(f"PUT /todos/{todo['id']}: {response.json()}")

response = client.delete(f"/todos/{todo['id']}")
print(f"DELETE /todos/{todo['id']}: status {response.status_code}")

response = client.get("/todos")
print(f"After delete: {response.json()}")
