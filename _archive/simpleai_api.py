from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import database

app = FastAPI(title="SimpleAI DB CRUD API", description="CRUD operations for simpleai.db")

# Initialize database connection
db = database.DataBase(db_path="simpleai.db")

# Pydantic models for Prompt
class PromptBase(BaseModel):
    prompt: str
    agent: Optional[str] = None
    model: Optional[str] = None
    response: Optional[str] = None
    abstract: Optional[str] = None
    should_end: int = 0

class PromptCreate(PromptBase):
    pass

class Prompt(PromptBase):
    id: int
    previous_id: Optional[int] = None

    class Config:
        orm_mode = True

# Pydantic models for Request
class RequestBase(BaseModel):
    prompt_id: int
    agent_name: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    success: Optional[int] = None
    include_history: int = 0

class RequestCreate(RequestBase):
    pass

class Request(RequestBase):
    id: int

    class Config:
        orm_mode = True

# Prompt Endpoints
@app.get("/prompts", response_model=List[Prompt])
def read_prompts(skip: int = 0, limit: int = 100):
    rows = db.prompts.Read()
    # Apply skip and limit manually since our Read doesn't support it directly
    # Convert rows to list of dicts
    prompts = []
    for row in rows[skip:skip+limit]:
        prompts.append({
            "id": row[0],
            "previous_id": row[1],
            "prompt": row[2],
            "agent": row[3],
            "model": row[4],
            "response": row[5],
            "abstract": row[6],
            "should_end": row[7]
        })
    return prompts

@app.get("/prompts/{prompt_id}", response_model=Prompt)
def read_prompt(prompt_id: int):
    rows = db.prompts.Read(condition=f"id={prompt_id}")
    if not rows:
        raise HTTPException(status_code=404, detail="Prompt not found")
    row = rows[0]
    return {
        "id": row[0],
        "previous_id": row[1],
        "prompt": row[2],
        "agent": row[3],
        "model": row[4],
        "response": row[5],
        "abstract": row[6],
        "should_end": row[7]
    }

@app.post("/prompts", response_model=Prompt)
def create_prompt(prompt: PromptCreate):
    # Insert into database
    prompt_id = db.prompts.Insert({
        "prompt": prompt.prompt,
        "agent": prompt.agent,
        "model": prompt.model,
        "response": prompt.response,
        "abstract": prompt.abstract,
        "should_end": prompt.should_end
    })
    # Get the inserted row
    rows = db.prompts.Read(condition=f"id={prompt_id}")
    row = rows[0]
    return {
        "id": row[0],
        "previous_id": row[1],
        "prompt": row[2],
        "agent": row[3],
        "model": row[4],
        "response": row[5],
        "abstract": row[6],
        "should_end": row[7]
    }

@app.put("/prompts/{prompt_id}", response_model=Prompt)
def update_prompt(prompt_id: int, prompt: PromptBase):
    # Check if exists
    existing = db.prompts.Read(condition=f"id={prompt_id}")
    if not existing:
        raise HTTPException(status_code=404, detail="Prompt not found")
    
    # Update
    db.prompts.Update({
        "prompt": prompt.prompt,
        "agent": prompt.agent,
        "model": prompt.model,
        "response": prompt.response,
        "abstract": prompt.abstract,
        "should_end": prompt.should_end
    }, condition=f"id={prompt_id}")
    
    # Get updated row
    rows = db.prompts.Read(condition=f"id={prompt_id}")
    row = rows[0]
    return {
        "id": row[0],
        "previous_id": row[1],
        "prompt": row[2],
        "agent": row[3],
        "model": row[4],
        "response": row[5],
        "abstract": row[6],
        "should_end": row[7]
    }

@app.delete("/prompts/{prompt_id}")
def delete_prompt(prompt_id: int):
    # Check if exists
    existing = db.prompts.Read(condition=f"id={prompt_id}")
    if not existing:
        raise HTTPException(status_code=404, detail="Prompt not found")
    
    # Delete
    db.prompts.db.cursor.execute(f"DELETE FROM Prompts WHERE id={prompt_id}")
    db.prompts.db.conn.commit()
    return {"message": "Prompt deleted successfully"}

# Request Endpoints
@app.get("/requests", response_model=List[Request])
def read_requests(skip: int = 0, limit: int = 100):
    rows = db.requests.Read()
    requests = []
    for row in rows[skip:skip+limit]:
        requests.append({
            "id": row[0],
            "prompt_id": row[1],
            "agent_name": row[2],
            "start_time": row[3],
            "end_time": row[4],
            "input_tokens": row[5],
            "output_tokens": row[6],
            "success": row[7],
            "include_history": row[8]
        })
    return requests

@app.get("/requests/{request_id}", response_model=Request)
def read_request(request_id: int):
    rows = db.requests.Read(condition=f"id={request_id}")
    if not rows:
        raise HTTPException(status_code=404, detail="Request not found")
    row = rows[0]
    return {
        "id": row[0],
        "prompt_id": row[1],
        "agent_name": row[2],
        "start_time": row[3],
        "end_time": row[4],
        "input_tokens": row[5],
        "output_tokens": row[6],
        "success": row[7],
        "include_history": row[8]
    }

@app.post("/requests", response_model=Request)
def create_request(request: RequestCreate):
    # Insert into database
    request_id = db.requests.Insert({
        "prompt_id": request.prompt_id,
        "agent_name": request.agent_name,
        "start_time": request.start_time,
        "end_time": request.end_time,
        "input_tokens": request.input_tokens,
        "output_tokens": request.output_tokens,
        "success": request.success,
        "include_history": request.include_history
    })
    # Get the inserted row
    rows = db.requests.Read(condition=f"id={request_id}")
    row = rows[0]
    return {
        "id": row[0],
        "prompt_id": row[1],
        "agent_name": row[2],
        "start_time": row[3],
        "end_time": row[4],
        "input_tokens": row[5],
        "output_tokens": row[6],
        "success": row[7],
        "include_history": row[8]
    }

@app.put("/requests/{request_id}", response_model=Request)
def update_request(request_id: int, request: RequestBase):
    # Check if exists
    existing = db.requests.Read(condition=f"id={request_id}")
    if not existing:
        raise HTTPException(status_code=404, detail="Request not found")
    
    # Update
    db.requests.Update({
        "prompt_id": request.prompt_id,
        "agent_name": request.agent_name,
        "start_time": request.start_time,
        "end_time": request.end_time,
        "input_tokens": request.input_tokens,
        "output_tokens": request.output_tokens,
        "success": request.success,
        "include_history": request.include_history
    }, condition=f"id={request_id}")
    
    # Get updated row
    rows = db.requests.Read(condition=f"id={request_id}")
    row = rows[0]
    return {
        "id": row[0],
        "prompt_id": row[1],
        "agent_name": row[2],
        "start_time": row[3],
        "end_time": row[4],
        "input_tokens": row[5],
        "output_tokens": row[6],
        "success": row[7],
        "include_history": row[8]
    }

@app.delete("/requests/{request_id}")
def delete_request(request_id: int):
    # Check if exists
    existing = db.requests.Read(condition=f"id={request_id}")
    if not existing:
        raise HTTPException(status_code=404, detail="Request not found")
    
    # Delete
    db.requests.db.cursor.execute(f"DELETE FROM Requests WHERE id={request_id}")
    db.requests.db.conn.commit()
    return {"message": "Request deleted successfully"}

# Health check
@app.get("/")
def root():
    return {"message": "SimpleAI DB CRUD API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)