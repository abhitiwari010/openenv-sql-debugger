from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict, Any

from server.environment import SQLEnvironment

app = FastAPI(title="SQL Debugger OpenEnv")
env = SQLEnvironment()

class ResetRequest(BaseModel):
    task_id: Optional[str] = None
    difficulty: Optional[str] = None

class StepRequest(BaseModel):
    action_type: str = "write_query"
    sql: str = ""
    explanation: Optional[str] = None
    metadata: dict = {}

class GradeRequest(BaseModel):
    task_id: str
    action: dict

@app.get("/")
def read_root():
    return {
        "info": "SQL Agent Environment API",
        "version": "1.0.0",
        "tasks": 6,
        "status": "running"
    }

@app.get("/health")
def read_health():
    return {"status": "healthy"}

@app.get("/state")
def read_state():
    return env.state

@app.post("/reset")
def do_reset(req: Optional[ResetRequest] = None):
    if req:
        obs = env.reset(task_id=req.task_id, difficulty=req.difficulty)
    else:
        obs = env.reset()
    return obs

@app.post("/step")
def do_step(req: StepRequest):
    obs = env.step(req.model_dump() if hasattr(req, "model_dump") else req.dict())
    return obs

@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "id": "easy_01",
                "name": "Filter high earners",
                "difficulty": "easy",
                "type": "write_query",
                "grader": True,
                "description": "Find all employees in Engineering with salary above 90000"
            },
            {
                "id": "easy_02",
                "name": "Count by department",
                "difficulty": "easy",
                "type": "write_query",
                "grader": True,
                "description": "Count employees per department ordered by count"
            },
            {
                "id": "medium_01",
                "name": "Fix broken JOIN",
                "difficulty": "medium",
                "type": "fix_query",
                "grader": True,
                "description": "Fix the broken JOIN query for active project assignments"
            },
            {
                "id": "medium_02",
                "name": "Fix wrong GROUP BY",
                "difficulty": "medium",
                "type": "fix_query",
                "grader": True,
                "description": "Fix the GROUP BY clause to aggregate by department"
            },
            {
                "id": "hard_01",
                "name": "Optimize correlated subquery",
                "difficulty": "hard",
                "type": "optimize_query",
                "grader": True,
                "description": "Replace correlated subqueries with window functions"
            },
            {
                "id": "hard_02",
                "name": "Eliminate N+1 problem",
                "difficulty": "hard",
                "type": "optimize_query",
                "grader": True,
                "description": "Rewrite N+1 query using LEFT JOIN and GROUP BY"
            }
        ]
    }

@app.post("/grade")
def do_grade(req: GradeRequest):
    env.reset(task_id=req.task_id)
    step_result = env.step(req.action)
    
    raw_score = step_result.get("reward", 0.05)
    score = max(0.05, min(0.95, float(raw_score)))
    score = round(score, 4)
    
    return {
        "task_id": req.task_id,
        "score": score,
        "feedback": step_result["observation"]["feedback"],
        "done": step_result["done"],
        "info": step_result.get("info", {})
    }
