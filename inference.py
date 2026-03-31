import os
import textwrap
from openai import OpenAI
from typing import List
from dotenv import load_dotenv

load_dotenv()

from client import SqlEnvClient
from models import SqlAction

# OpenEnv execution environment config
ENV_CONTAINER_START = os.getenv("ENV_CONTAINER_START", "false").lower() == "true"
ENV_IMAGE_NAME = "sql-agent-env:latest"

# LLM inference config as per instructions
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
MAX_STEPS = 10

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
    if not API_KEY and not GEMINI_API_KEY:
        print("WARNING: API credentials might be missing. Using dummy environment flow for validation.")
    
    # Client setup
    client_type = "openai"
    if GEMINI_API_KEY:
        try:
            from google import genai
            client = genai.Client(api_key=GEMINI_API_KEY)
            client_type = "gemini"
            print("--- Using Google Gemini Client ---")
        except ImportError:
            print("WARNING: google-genai package not found. Falling back to OpenAI client.")
            client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "dummy-key")
    else:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "dummy-key")

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

    history = []
    
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
            
            try:
                if client_type == "gemini":
                    # For Gemini, we collapse System & User prompts into a single payload
                    gemini_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
                    # Use provided model Name, fallback defaults to gemini-2.5-flash
                    gemini_model = MODEL_NAME if MODEL_NAME != "gpt-4o-mini" else "gemini-2.5-flash"
                    
                    response = client.models.generate_content(
                        model=gemini_model,
                        contents=gemini_prompt,
                        config={"temperature": 0.1}
                    )
                    response_text = response.text or "SELECT * FROM DEPARTMENTS"
                elif API_KEY or os.getenv("OPENAI_API_KEY"):
                    completion = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        temperature=0.1,
                        stream=False
                    )
                    response_text = completion.choices[0].message.content or "SELECT * FROM DEPARTMENTS"
                else:
                    response_text = "SELECT * FROM DEPARTMENTS"  # Dummy mode

            except Exception as e:
                print(f"Model Request failed: {e}. Using fallback action.")
                response_text = "SELECT 1"

            action_str = parse_model_action(response_text)
            print(f"\n[Step {step}] Model suggested: {action_str}")
            
            result = env.step(SqlAction(query=action_str))
            observation = result.observation
            reward = result.reward
            done = result.done

            print(f"  Reward: {reward:+.2f} | Done: {done}")
            if observation.execution_error:
                print(f"  Error: {observation.execution_error[:200]}")
                
            history.append(f"Q: {action_str} -> R: {reward:+.2f}")
            
            if done:
                print("\nAll tasks complete!")
                break
                
    finally:
        env.close()

if __name__ == "__main__":
    main()
