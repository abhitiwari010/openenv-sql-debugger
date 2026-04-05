import uuid

from openenv.core.env_server import Environment
from models import SqlAction, SqlObservation, SqlState, SqlReward
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
        self._dangerous_keywords = ("drop ", "delete ", "truncate ", "alter ", "update ", "insert ")

    def reset(self) -> SqlObservation:
        self._state = SqlState(
            episode_id=str(uuid.uuid4()),
            current_task_index=0,
            total_tasks=self.task_registry.get_total_tasks(),
            accumulated_reward=0.0,
            task_scores=[],
            attempts_per_task=[0 for _ in range(self.task_registry.get_total_tasks())],
        )
        first_task = self.task_registry.get_task(0)
        return SqlObservation(
            current_task_instruction=first_task.instruction,
            schema_info=self.db.get_schema_summary(),
            task_id=first_task.task_id,
            difficulty=first_task.difficulty_label,
        )

    @property
    def state(self) -> SqlState:
        return self._state

    def step(self, action: SqlAction) -> SqlObservation:
        self._state.step_count += 1
        query = action.query.strip()
        current_idx = self._state.current_task_index
        n_tasks = self.task_registry.get_total_tasks()
        while len(self._state.attempts_per_task) < n_tasks:
            self._state.attempts_per_task.append(0)
        task = self.task_registry.get_task(current_idx)
        self._state.attempts_per_task[current_idx] += 1
        
        execution_result_str = None
        error_str = None
        score = 0.0
        grader_feedback = ""
        reward = SqlReward(total=0.0, task_score=0.0)
        done = False

        query_lower = query.lower()
        if any(keyword in query_lower for keyword in self._dangerous_keywords):
            reward.safety_penalty = -1.0
            reward.total += reward.safety_penalty
            error_str = "Destructive SQL command detected. Only read-only SELECT queries are allowed."
        else:
            reward.step_penalty = -0.02
            reward.total += reward.step_penalty
        
        try:
            if error_str:
                raise ValueError(error_str)

            # 1. Gather expected vs agent dataframes
            expected_df = self.db.execute_dataframe(task.expected_query)
            agent_df = self.db.execute_dataframe(query)
            
            # Serialize for observation
            execution_result_str = agent_df.to_json(orient="records")
            
            # 2. Grade execution
            grade_result = DataFrameGrader.grade_with_details(agent_df, expected_df)
            score = grade_result["score"]
            grader_feedback = grade_result["feedback"]
            reward.details = grade_result["details"]
            
            # 3. Reward shaping
            reward.task_score = score
            reward.valid_sql_bonus = 0.1
            reward.progress_bonus = 0.9 * score
            reward.total += reward.valid_sql_bonus + reward.progress_bonus
            
            # 4. Check level progression
            if score >= 0.95:
                self._state.task_scores.append(score)
                self._state.current_task_index += 1
                if self._state.current_task_index >= self.task_registry.get_total_tasks():
                    done = True
            elif self._state.attempts_per_task[current_idx] >= self._state.max_attempts_per_task:
                # Episode ends if agent gets stuck on same task
                self._state.task_scores.append(score)
                done = True
                     
        except Exception as e:
            error_str = str(e)
            reward.error_penalty = -0.5
            reward.total += reward.error_penalty
        
        # Global accumulation
        self._state.last_reward = reward.total
        reward_breakdown = reward.model_dump() if hasattr(reward, "model_dump") else reward.dict()
        self._state.last_info = {
            "task_id": task.task_id,
            "difficulty": task.difficulty_label,
            "task_score": score,
            "attempts_for_task": self._state.attempts_per_task[current_idx],
            "reward_breakdown": reward_breakdown,
        }
        self._state.accumulated_reward += reward.total
        
        # Prepare observation response
        if done:
            total_score = sum(self._state.task_scores)
            next_instruction = f"All tasks completed! Final SQL Capability Score: {total_score}"
            schema = ""
            task_id = task.task_id
            difficulty = task.difficulty_label
        else:
            next_task = self.task_registry.get_task(self._state.current_task_index)
            next_instruction = next_task.instruction
            schema = self.db.get_schema_summary()
            task_id = next_task.task_id
            difficulty = next_task.difficulty_label
            
        obs = SqlObservation(
            current_task_instruction=next_instruction,
            schema_info=schema,
            task_id=task_id,
            difficulty=difficulty,
            execution_result=execution_result_str,
            execution_error=error_str,
            task_score=score,
            grader_feedback=grader_feedback,
            reward=reward.total,
            done=done,
        )

        return obs
