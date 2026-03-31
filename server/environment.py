import uuid
import pandas as pd
from typing import Tuple

from openenv.core.env_server import Environment
from models import SqlAction, SqlObservation, SqlState
from database import SqliteManager
from core import EnvironmentTasks, DataFrameGrader

class SqlEnvironment(Environment):
    """Orchestrates the SQL Agent execution matching the OpenEnv Gym API."""
    def __init__(self):
        super().__init__()
        self._state = SqlState()
        self.task_registry = EnvironmentTasks()
        
        # Instantiate db connection immediately for the server loop
        self.db = SqliteManager()
        self.db.connect()

    def reset(self) -> SqlObservation:
        self._state = SqlState(
            episode_id=str(uuid.uuid4()),
            current_task_index=0,
            total_tasks=self.task_registry.get_total_tasks(),
            accumulated_reward=0.0,
            task_scores=[]
        )
        first_task = self.task_registry.get_task(0)
        return SqlObservation(
            current_task_instruction=first_task.instruction,
            schema_info=self.db.get_schema_summary()
        )

    def state(self) -> SqlState:
        return self._state

    def step(self, action: SqlAction) -> Tuple[SqlObservation, float, bool]:
        self._state.step_count += 1
        query = action.query
        current_idx = self._state.current_task_index
        task = self.task_registry.get_task(current_idx)
        
        execution_result_str = None
        error_str = None
        reward = 0.0
        done = False
        
        try:
            # 1. Gather expected vs agent dataframes
            expected_df = self.db.execute_dataframe(task.expected_query)
            agent_df = self.db.execute_dataframe(query)
            
            # Serialize for observation
            execution_result_str = agent_df.to_json(orient="records")
            
            # 2. Grade execution
            score = DataFrameGrader.grade(agent_df, expected_df)
            self._state.task_scores.append(score)
            
            # 3. Reward scaling
            reward += 0.1 # Valid execution baseline
            reward += score
            
            # 4. Check level progression
            if score == 1.0:
                 self._state.current_task_index += 1
                 if self._state.current_task_index >= self.task_registry.get_total_tasks():
                     done = True
                     
        except Exception as e:
            error_str = str(e)
            reward -= 0.5  # Heavy penalty for destructive/invalid SQL
        
        # Global accumulation
        self._state.accumulated_reward += reward
        
        # Prepare observation response
        if done:
            total_score = sum(self._state.task_scores)
            next_instruction = f"All tasks completed! Final SQL Capability Score: {total_score}"
            schema = ""
        else:
            next_task = self.task_registry.get_task(self._state.current_task_index)
            next_instruction = next_task.instruction
            schema = self.db.get_schema_summary()
            
        obs = SqlObservation(
            current_task_instruction=next_instruction,
            schema_info=schema,
            execution_result=execution_result_str,
            execution_error=error_str
        )
        
        return obs, reward, done
