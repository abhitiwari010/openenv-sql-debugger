from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from openenv.core.env_server import Action, Observation, State

class SqlAction(Action):
    """Action representing a SQL query execution."""
    query: str

class SqlObservation(Observation):
    """Observation returned after executing a SQL query (or at reset)."""
    current_task_instruction: str
    schema_info: str
    task_id: str
    difficulty: str
    execution_result: Optional[str] = None
    execution_error: Optional[str] = None
    task_score: float = 0.0
    grader_feedback: Optional[str] = None


class SqlReward(BaseModel):
    """Structured reward payload for deterministic scoring details."""
    total: float
    task_score: float
    valid_sql_bonus: float = 0.0
    progress_bonus: float = 0.0
    step_penalty: float = 0.0
    safety_penalty: float = 0.0
    error_penalty: float = 0.0
    details: Dict[str, Any] = Field(default_factory=dict)

class SqlState(State):
    """State tracking for the SQL agent environment."""
    current_task_index: int = 0
    total_tasks: int = 3
    accumulated_reward: float = 0.0
    task_scores: List[float] = Field(default_factory=list)
    attempts_per_task: List[int] = Field(default_factory=list)
    max_attempts_per_task: int = 6
    last_reward: float = 0.0
    last_info: Dict[str, Any] = Field(default_factory=dict)
