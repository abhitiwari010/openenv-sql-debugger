from pydantic import Field
from typing import Optional, List
from openenv.core.env_server import Action, Observation, State

class SqlAction(Action):
    """Action representing a SQL query execution."""
    query: str

class SqlObservation(Observation):
    """Observation returned after executing a SQL query (or at reset)."""
    current_task_instruction: str
    schema_info: str
    execution_result: Optional[str] = None
    execution_error: Optional[str] = None

class SqlState(State):
    """State tracking for the SQL agent environment."""
    current_task_index: int = 0
    total_tasks: int = 3
    accumulated_reward: float = 0.0
    task_scores: List[float] = Field(default_factory=list)
