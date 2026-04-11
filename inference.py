import os
import re
import json
import requests
from openai import OpenAI
import sys

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
ENV_NAME = "sql_debugger"
MAX_STEPS = 6
TEMPERATURE = 0.1
MAX_TOKENS = 500

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

SYSTEM_PROMPT = """You are an expert SQL developer working with a SQLite database.

Your job depends on the task type:
- write_query: Write a correct SQL SELECT query from scratch
- fix_query: You are shown broken SQL. Fix it so it runs correctly
- optimize_query: You are shown slow SQL. Rewrite it using CTEs (WITH clause) 
  or window functions (ROW_NUMBER, RANK) instead of correlated subqueries

RESPONSE FORMAT — always respond with ONLY this JSON:
{"action_type": "write_query", "sql": "SELECT ...", "explanation": "brief reason"}

RULES:
- Only SELECT statements are allowed
- No DROP, DELETE, INSERT, UPDATE, CREATE, ALTER, TRUNCATE
- For fix_query: keep the same intent, just fix the bugs
- For optimize_query: use WITH clause or window functions
- Respond ONLY with the JSON object, no other text"""

def build_prompt(obs: dict, step: int, history: list) -> str:
    parts = []
    parts.append(f"=== TASK ===\n{obs.get('task_description', '')}")
    
    schema = obs.get('schema_info', '')
    if schema:
        parts.append(f"=== DATABASE SCHEMA ===\n{chr(10).join(schema.split(chr(10))[:30])}")
        
    data = obs.get('sample_data', '')
    if data:
        parts.append(f"=== SAMPLE DATA ===\n{chr(10).join(data.split(chr(10))[:20])}")
        
    hints = obs.get('expected_description', '')
    if hints:
        parts.append(f"=== HINTS ===\n{hints}")
        
    last_sql = obs.get('last_sql')
    if last_sql:
        parts.append(f"=== YOUR PREVIOUS SQL ===\n{last_sql}")
        
    last_result = obs.get('last_result')
    if last_result:
        parts.append(f"=== RESULT OF PREVIOUS SQL ===\n{last_result}")
        
    last_error = obs.get('last_error')
    if last_error:
        parts.append(f"=== ERROR ===\n{last_error}")
        
    feedback = obs.get('feedback', '')
    if feedback and step > 1:
        parts.append(f"=== GRADER FEEDBACK ===\n{feedback}")
        
    if history:
        hist_str = "\n".join(history[-3:])
        parts.append(f"=== RECENT HISTORY ===\n{hist_str}")
        
    parts.append("Respond with ONLY a JSON action object.")
    return "\n\n".join(parts)

def parse_action(response_text: str, task_type: str) -> dict:
    text = response_text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
        
    match = re.search(r'\{[^{}]+\}', text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
            
    sql_match = re.search(r'(?:SELECT|WITH).+', text, re.DOTALL | re.IGNORECASE)
    if sql_match:
        sql = sql_match.group(0).strip()
        if sql.endswith("```"): sql = sql[:-3].strip()
        return {"action_type": task_type, "sql": sql, "explanation": "Regex parsed"}
        
    return {"action_type": task_type, "sql": "SELECT 1", "explanation": "parse failed"}

def run_episode(task_id: str):
    try:
        res = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id, "difficulty": None})
        res.raise_for_status()
        obs_obj = res.json()
        obs = obs_obj["observation"]
    except Exception as e:
        # Failsafe reset
        obs = {"task_type": "write_query"}
        
    print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}", flush=True)
    
    best_reward = 0.05
    history = []
    
    step = 0
    done = False
    
    for step in range(1, MAX_STEPS + 1):
        if obs.get("done", False):
            break
            
        prompt = build_prompt(obs, step, history)
        
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as e:
            response_text = ""
            
        action = parse_action(response_text, obs.get("task_type", "write_query"))
        
        # Safe JSON stringification for the action output in the log
        action_str = json.dumps(action)
        
        try:
            step_res = requests.post(f"{ENV_URL}/step", json=action)
            step_res.raise_for_status()
            step_data = step_res.json()
        except Exception as e:
            # Fallback if step errors
            step_data = {"reward": 0.05, "observation": {"done": True}, "done": True, "error": str(e)}
            
        reward = step_data.get("reward", 0.05)
        done = step_data.get("done", False)
        err = step_data.get("error")
        # Format the numbers cleanly
        reward_formatted = f"{reward:.2f}"
        
        best_reward = max(best_reward, reward)
        obs = step_data.get("observation", {})
        
        # Convert flags to lowercase 'true' or 'false' for validator tracking
        done_str = "true" if done else "false"
        err_str = "null" if not err else f'"{str(err)}"'
        
        print(f"[STEP] step={step} action={action_str} reward={reward_formatted} done={done_str} error={err_str}", flush=True)
        
        history.append(f"Step {step}: reward={reward_formatted}")
        
        if done:
            break
            
    # Track final outcome: successes usually represent the max reward crossing some threshold 
    success_str = "true" if best_reward > 0.8 else "false"
    best_reward_formatted = f"{best_reward:.2f}"
    
    print(f"[END] task={task_id} success={success_str} steps={step} score={best_reward_formatted}", flush=True)

def main():
    # Only process the tasks quietly!
    task_ids = ["easy_01", "easy_02", "medium_01", "medium_02", "hard_01", "hard_02"]
    for tid in task_ids:
        run_episode(tid)

if __name__ == "__main__":
    main()
