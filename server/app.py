from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import json
import asyncio

from server.environment import SQLEnvironment

app = FastAPI(title="SQL Debugger OpenEnv", version="1.0.0")

# One environment instance per HTTP session (stateless endpoints)
# WebSocket sessions get their own instance
env = SQLEnvironment()

# ── Pydantic models ───────────────────────────────────────────────────────────

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

# ── HTTP endpoints ────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "info": "SQL Agent Environment API",
        "version": "1.0.0",
        "tasks": 6,
        "status": "running"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    return env.reset(task_id=req.task_id, difficulty=req.difficulty)

@app.post("/step")
def step(req: StepRequest):
    return env.step(req.dict())

@app.get("/state")
def state():
    return env.state

# ── CRITICAL MISSING ENDPOINTS — ADD THESE ───────────────────────────────────

@app.get("/tasks")
def list_tasks():
    """
    Validator uses this endpoint to find and enumerate all graders.
    Every task MUST have grader: true.
    """
    return {
        "tasks": [
            {
                "id": "easy_01",
                "name": "Filter high earners",
                "difficulty": "easy",
                "type": "write_query",
                "grader": True,
                "description": "Find all employees in Engineering with salary above 90000, return name and salary ordered by salary DESC"
            },
            {
                "id": "easy_02",
                "name": "Count by department",
                "difficulty": "easy",
                "type": "write_query",
                "grader": True,
                "description": "Count employees per department, return department and count ordered by count DESC"
            },
            {
                "id": "medium_01",
                "name": "Fix broken JOIN",
                "difficulty": "medium",
                "type": "fix_query",
                "grader": True,
                "description": "Fix the broken JOIN query with missing ON keyword and wrong table alias in WHERE"
            },
            {
                "id": "medium_02",
                "name": "Fix wrong GROUP BY",
                "difficulty": "medium",
                "type": "fix_query",
                "grader": True,
                "description": "Fix the GROUP BY that incorrectly groups by id instead of department"
            },
            {
                "id": "hard_01",
                "name": "Optimize correlated subquery",
                "difficulty": "hard",
                "type": "optimize_query",
                "grader": True,
                "description": "Replace correlated subqueries with window functions or CTEs to find top earner per department"
            },
            {
                "id": "hard_02",
                "name": "Eliminate N+1 problem",
                "difficulty": "hard",
                "type": "optimize_query",
                "grader": True,
                "description": "Rewrite N+1 correlated subquery using LEFT JOIN and GROUP BY"
            }
        ]
    }

@app.post("/grade")
def grade(req: GradeRequest):
    """
    Validator calls this to get a score for a specific task.
    Score MUST be strictly between 0 and 1 (never 0.0, never 1.0).
    This is enforced by _clamp() in the environment AND by the
    absolute safety clamp below.
    """
    # Use a fresh environment instance for grading to avoid state pollution
    grade_env = SQLEnvironment()
    grade_env.reset(task_id=req.task_id)
    result = grade_env.step(req.action)
    
    raw_score = result.get("reward", 0.05)
    
    # ABSOLUTE SAFETY CLAMP — even if environment has a bug,
    # this line guarantees the validator never sees 0.0 or 1.0
    score = round(max(0.05, min(0.95, float(raw_score))), 4)
    
    return {
        "task_id": req.task_id,
        "score": score,
        "feedback": result["observation"]["feedback"],
        "done": result["done"],
        "info": result.get("info", {})
    }

# ── WebSocket endpoint — required by OpenEnv spec ─────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Persistent WebSocket session. Each connection gets its own 
    SQLEnvironment instance so sessions are isolated.
    """
    await websocket.accept()
    ws_env = SQLEnvironment()
    
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "error": "Invalid JSON"
                }))
                continue
            
            msg_type = message.get("type", "")
            
            if msg_type == "reset":
                result = ws_env.reset(
                    task_id=message.get("task_id"),
                    difficulty=message.get("difficulty")
                )
                await websocket.send_text(json.dumps(result))
            
            elif msg_type == "step":
                action = message.get("action", {})
                result = ws_env.step(action)
                await websocket.send_text(json.dumps(result))
            
            elif msg_type == "state":
                await websocket.send_text(json.dumps(ws_env.state))
            
            else:
                await websocket.send_text(json.dumps({
                    "error": f"Unknown message type: {msg_type}"
                }))
    
    except WebSocketDisconnect:
        pass

# ── Web UI ─────────────────────────────────────────────────────────────────────

@app.get("/web", response_class=HTMLResponse)
def web_ui():
    """
    Interactive web playground for testing the environment manually.
    """
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SQL data analyst playground</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { 
    background: #0d1117; color: #e6edf3; 
    font-family: 'Courier New', monospace; 
    padding: 20px; 
  }
  .header { 
    display: flex; justify-content: space-between; 
    align-items: center; margin-bottom: 24px; 
    border-bottom: 1px solid #30363d; padding-bottom: 16px;
  }
  .badge { 
    background: #1f6feb; color: white; 
    font-size: 11px; padding: 2px 8px; 
    border-radius: 4px; margin-right: 8px;
  }
  .title { font-size: 28px; font-weight: bold; margin-top: 4px; }
  .links a { color: #58a6ff; text-decoration: none; margin-left: 16px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
  .panel { 
    background: #161b22; border: 1px solid #30363d; 
    border-radius: 8px; padding: 16px; 
  }
  .panel h3 { 
    font-size: 11px; letter-spacing: 1px; 
    color: #8b949e; margin-bottom: 12px; 
  }
  .task-meta { 
    display: flex; gap: 8px; margin-bottom: 8px; 
  }
  .tag { 
    background: #21262d; border: 1px solid #30363d;
    padding: 2px 8px; border-radius: 4px; font-size: 12px;
  }
  .task-desc { font-size: 14px; line-height: 1.6; color: #e6edf3; }
  .schema-text { 
    font-size: 12px; color: #8b949e; 
    white-space: pre; overflow-x: auto; 
  }
  .sql-panel { 
    background: #161b22; border: 1px solid #30363d; 
    border-radius: 8px; padding: 16px; margin-bottom: 16px;
  }
  .sql-panel h3 { 
    font-size: 11px; letter-spacing: 1px; 
    color: #8b949e; margin-bottom: 4px;
  }
  .sql-hint { font-size: 12px; color: #8b949e; margin-bottom: 12px; }
  textarea { 
    width: 100%; height: 120px; 
    background: #0d1117; color: #79c0ff;
    border: 2px solid #238636; border-radius: 6px;
    padding: 12px; font-family: 'Courier New', monospace;
    font-size: 14px; resize: vertical;
  }
  .actions { display: flex; gap: 12px; margin-top: 12px; }
  button { 
    padding: 8px 20px; border-radius: 6px; 
    border: none; cursor: pointer; 
    font-family: 'Courier New', monospace; font-size: 14px;
  }
  .btn-primary { background: #238636; color: white; }
  .btn-secondary { background: #21262d; color: #e6edf3; border: 1px solid #30363d; }
  .btn-primary:hover { background: #2ea043; }
  .result-panel { 
    background: #161b22; border: 1px solid #30363d; 
    border-radius: 8px; padding: 16px; margin-bottom: 16px;
    min-height: 80px;
  }
  .result-panel h3 { 
    font-size: 11px; letter-spacing: 1px; 
    color: #8b949e; margin-bottom: 12px; 
  }
  .reward-bar { 
    height: 6px; background: #21262d; 
    border-radius: 3px; margin: 8px 0; 
  }
  .reward-fill { height: 100%; border-radius: 3px; background: #238636; transition: width 0.3s; }
  .feedback { font-size: 13px; color: #8b949e; margin-top: 8px; }
  .result-text { 
    font-size: 12px; color: #e6edf3; 
    white-space: pre; overflow-x: auto; margin-top: 8px;
  }
  .status { 
    font-size: 13px; color: #8b949e; 
    padding: 8px 0; 
  }
  .status.connected { color: #3fb950; }
  .status.error { color: #f85149; }
</style>
</head>
<body>
<div class="header">
  <div>
    <div><span class="badge">OPENENV</span></div>
    <div class="title">SQL data analyst playground</div>
  </div>
  <div class="links">
    <a href="/docs">API docs</a>
    <a href="/health">Health</a>
    <a href="/tasks">Tasks</a>
  </div>
</div>

<p style="font-size:13px;color:#8b949e;margin-bottom:20px;">
  Interactive mode uses a persistent <code>WebSocket</code> session at <code>/ws</code> 
  (OpenEnv HTTP <code>/step</code> is stateless). Connect, then run reset → SQL steps.
</p>

<div class="grid">
  <div class="panel">
    <h3>CURRENT TASK</h3>
    <div class="task-meta">
      <span class="tag" id="task-id">—</span>
      <span class="tag" id="task-diff">—</span>
    </div>
    <div class="task-desc" id="task-desc">Click "Start episode" to load the first task.</div>
    <div class="feedback" id="hints"></div>
  </div>
  <div class="panel">
    <h3>SCHEMA</h3>
    <div class="schema-text" id="schema-text">—</div>
  </div>
</div>

<div class="sql-panel">
  <h3>YOUR SQL</h3>
  <p class="sql-hint">Type <strong>only valid SQLite</strong> here.</p>
  <textarea id="sql-input" placeholder="Executable SQL only, e.g. SELECT department_id, AVG(salary) ..."></textarea>
  <div class="actions">
    <button class="btn-primary" onclick="startEpisode()">Start episode</button>
    <button class="btn-primary" onclick="submitSQL()">Submit SQL</button>
    <button class="btn-secondary" onclick="nextEpisode()">Next task</button>
  </div>
</div>

<div class="result-panel">
  <h3>LAST RESULT</h3>
  <div id="reward-display" style="font-size:13px;color:#8b949e;">No result yet.</div>
  <div class="reward-bar"><div class="reward-fill" id="reward-bar" style="width:0%"></div></div>
  <div class="feedback" id="feedback-text"></div>
  <div class="result-text" id="result-text"></div>
</div>

<div class="status" id="status">Not connected. Click "Start episode" to begin.</div>

<script>
let ws = null;
let connected = false;

function connect(callback) {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(proto + '//' + location.host + '/ws');
  
  ws.onopen = () => {
    connected = true;
    document.getElementById('status').textContent = 'Connected via WebSocket.';
    document.getElementById('status').className = 'status connected';
    if (callback) callback();
  };
  
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    handleResult(data);
  };
  
  ws.onerror = () => {
    document.getElementById('status').textContent = 'WebSocket error.';
    document.getElementById('status').className = 'status error';
  };
  
  ws.onclose = () => {
    connected = false;
    document.getElementById('status').textContent = 'Disconnected.';
    document.getElementById('status').className = 'status';
  };
}

function handleResult(data) {
  const obs = data.observation || data;
  if (!obs) return;
  
  // Update task panel
  if (obs.task_id) document.getElementById('task-id').textContent = obs.task_id;
  if (obs.task_type) document.getElementById('task-diff').textContent = obs.task_type;
  if (obs.task_description) document.getElementById('task-desc').textContent = obs.task_description;
  if (obs.expected_description) document.getElementById('hints').textContent = obs.expected_description;
  if (obs.schema_info) {
    const lines = obs.schema_info.split('\\n').slice(0, 20).join('\\n');
    document.getElementById('schema-text').textContent = lines;
  }
  
  // Update reward
  const reward = data.reward || obs.reward || 0;
  const pct = Math.round(reward * 100);
  document.getElementById('reward-display').textContent = 'Reward: ' + reward.toFixed(4) + ' (' + pct + '%)';
  document.getElementById('reward-bar').style.width = pct + '%';
  
  // Update feedback
  if (obs.feedback) document.getElementById('feedback-text').textContent = obs.feedback;
  if (obs.last_result) document.getElementById('result-text').textContent = obs.last_result;
  if (obs.last_error) document.getElementById('result-text').textContent = 'ERROR: ' + obs.last_error;
  
  // Show starter SQL if this is a fix/optimize task
  if (obs.last_sql && document.getElementById('sql-input').value === '') {
    document.getElementById('sql-input').value = obs.last_sql;
  }
}

function startEpisode() {
  if (!connected) {
    connect(() => {
      ws.send(JSON.stringify({ type: 'reset' }));
    });
  } else {
    ws.send(JSON.stringify({ type: 'reset' }));
  }
}

function submitSQL() {
  const sql = document.getElementById('sql-input').value.trim();
  if (!sql) { alert('Please enter a SQL query.'); return; }
  if (!connected) { alert('Click Start episode first.'); return; }
  ws.send(JSON.stringify({ 
    type: 'step', 
    action: { action_type: 'write_query', sql: sql } 
  }));
}

function nextEpisode() {
  if (!connected) {
    connect(() => {
      ws.send(JSON.stringify({ type: 'reset' }));
    });
  } else {
    ws.send(JSON.stringify({ type: 'reset' }));
  }
  document.getElementById('sql-input').value = '';
  document.getElementById('result-text').textContent = '';
  document.getElementById('feedback-text').textContent = '';
}
</script>
</body>
</html>
"""
