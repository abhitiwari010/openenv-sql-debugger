from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional, Dict, Any

from server.environment import SqlEnvironment, TASKS

app = FastAPI(title="SQL Agent Environment API")

env = SqlEnvironment()

class ResetRequest(BaseModel):
    task_id: Optional[str] = None
    difficulty: Optional[str] = None

class ActionRequest(BaseModel):
    sql: str
    action_type: Optional[str] = None

class GradeRequest(BaseModel):
    task_id: str
    action: ActionRequest

@app.get("/")
def read_root():
    return {"info": "SQL Agent Environment API"}

@app.get("/health")
def read_health():
    return {"status": "healthy"}

@app.get("/state")
def read_state():
    return env.state

@app.get("/tasks")
def list_tasks():
    return {"tasks": TASKS}

@app.post("/reset")
def do_reset(req: Optional[ResetRequest] = None):
    if req:
        obs = env.reset(task_id=req.task_id, difficulty=req.difficulty)
    else:
        obs = env.reset()
    return obs

@app.post("/step")
def do_step(action: ActionRequest):
    obs = env.step({"sql": action.sql, "action_type": action.action_type})
    return obs

@app.post("/grade")
def do_grade(req: GradeRequest):
    env.reset(task_id=req.task_id)
    obs = env.step({"sql": req.action.sql, "action_type": req.action.action_type})
    score = max(0.05, min(0.95, float(obs["reward"])))
    
    return {
        "task_id": req.task_id,
        "score": score,
        "feedback": obs["feedback"],
        "done": obs["done"]
    }
