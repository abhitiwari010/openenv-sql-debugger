import os
import textwrap
from openai import OpenAI
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

from client import SqlEnvClient
from models import SqlAction

# OpenEnv execution environment config
ENV_CONTAINER_START = os.getenv("ENV_CONTAINER_START", "false").lower() == "true"
ENV_IMAGE_NAME = "sql-agent-env:latest"

# LLM inference config as per instructions
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
MAX_STEPS = int(os.getenv("MAX_STEPS", "20"))
OPENAI_SEED = int(os.getenv("OPENAI_SEED", "42"))

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert Data Analyst and SQL Agent.
    You will be provided with a SQL Schema and an instruction.
    Reply with exactly one JSON block containing the SQL query to execute.
    Do NOT provide markdown formatting or explanations. DO NOT wrap in ```sql.
    The response should be the pure string of the query itself.
    """
).strip()

def build_user_prompt(step: int, observation, history: List[str]) -> str:
    schema = observation.schema_info
    instruction = observation.current_task_instruction
    exec_result = observation.execution_result or "None"
    error = observation.execution_error or "None"
    
    prompt = textwrap.dedent(
        f"""
        Step: {step}
        Database Schema:
        {schema}
        
        Task Instruction:
        {instruction}
        
        Previous Execution Result: {exec_result}
        Previous Error: {error}
        
        Write the precise SQL query string to accomplish the task instruction. Return nothing else.
        """
    ).strip()
    return prompt

def parse_model_action(response_text: str) -> str:
    query = response_text.strip()
    if query.startswith("```sql"):
        query = query[6:]
    if query.startswith("```"):
        query = query[3:]
    if query.endswith("```"):
        query = query[:-3]
    return query.strip()

def main():
    if not API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required for reproducible baseline runs.")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print("--- SQL Data Analyst OpenEnv Baseline ---")
    
    if ENV_CONTAINER_START:
        print(f"Starting docker image: {ENV_IMAGE_NAME}")
        env = SqlEnvClient.from_docker_image(ENV_IMAGE_NAME).sync()
    else:
        # Fallback to local memory-bound initialization
        from server.environment import SqlEnvironment
        base_env = SqlEnvironment()
        
        # Direct wrapper
        class DirectClient:
            def __init__(self, target):
                self.target = target
            def reset(self):
                obs = self.target.reset()
                return type('StepResult', (), {'observation': obs, 'reward': 0.0, 'done': False})()
            def step(self, action):
                obs, reward, done = self.target.step(action)
                return type('StepResult', (), {'observation': obs, 'reward': reward, 'done': done})()
            def close(self):
                pass
        env = DirectClient(base_env)

    history: List[str] = []
    task_scores: Dict[str, float] = {}
    run_trace: List[Dict[str, Any]] = []
    
    try:
        result = env.reset()
        observation = result.observation
        print(f"\n[Episode Start] Initial Task: {observation.current_task_instruction}")
        
        for step in range(1, MAX_STEPS + 1):
            if result.done:
                print("Environment signalled done. Stopping early.")
                break
                
            user_prompt = build_user_prompt(step, observation, history)
            
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0,
                seed=OPENAI_SEED,
                stream=False,
            )
            response_text = completion.choices[0].message.content or "SELECT 1"

            action_str = parse_model_action(response_text)
            print(f"\n[Step {step}] Model suggested: {action_str}")
            current_task_id = observation.task_id
            result = env.step(SqlAction(query=action_str))
            observation = result.observation
            reward = result.reward
            done = result.done

            print(f"  Reward: {reward:+.2f} | Done: {done}")
            print(f"  Task Score: {observation.task_score:.3f} | Difficulty: {observation.difficulty}")
            if observation.grader_feedback:
                print(f"  Grader: {observation.grader_feedback}")
            if observation.execution_error:
                print(f"  Error: {observation.execution_error[:200]}")

            if observation.task_score > 0:
                task_scores[current_task_id] = max(task_scores.get(current_task_id, 0.0), observation.task_score)
                
            history.append(f"Q: {action_str} -> R: {reward:+.2f}")
            run_trace.append(
                {
                    "step": step,
                    "task_id": current_task_id,
                    "action": action_str,
                    "reward": reward,
                    "task_score": observation.task_score,
                    "done": done,
                }
            )
            
            if done:
                print("\nAll tasks complete!")
                break

        print("\n=== Baseline Summary ===")
        for task_id in sorted(task_scores):
            print(f"{task_id}: {task_scores[task_id]:.3f}")
        task_avg = (sum(task_scores.values()) / len(task_scores)) if task_scores else 0.0
        total_return = sum(item["reward"] for item in run_trace)
        print(f"average_task_score: {task_avg:.3f}")
        print(f"episode_return: {total_return:.3f}")
        print(f"model: {MODEL_NAME} | seed: {OPENAI_SEED}")
        print("reproducibility_note: deterministic prompt, temperature=0.0, fixed seed")

    finally:
        env.close()

if __name__ == "__main__":
    main()
