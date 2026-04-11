import os
import re
import json
import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000") # defaults to common uvicorn port but user specified 7860
MAX_STEPS = 6
TEMPERATURE = 0.1
MAX_TOKENS = 500

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

SYSTEM_PROMPT = """You are an expert SQL developer.
For write_query tasks: write a correct SQL SELECT query.
For fix_query tasks: fix the broken SQL provided.
For optimize_query tasks: rewrite slow SQL using CTEs or JOINs instead of subqueries.

ALWAYS respond with ONLY a JSON object like this:
{"action_type": "write_query", "sql": "SELECT ...", "explanation": "reason"}

Rules: Only SELECT allowed. No DROP DELETE INSERT UPDATE CREATE ALTER."""

def build_prompt(obs: dict) -> str:
    obs_data = obs.get("observation", obs)
    task_desc = obs_data.get("task_description", "")
    schema_info = obs_data.get("schema_info", "")
    sample_data = obs_data.get("sample_data", "")
    hints = obs_data.get("metadata", [])
    last_sql = obs_data.get("last_sql", "")
    last_result = obs_data.get("last_result", "")
    last_error = obs_data.get("last_error", "")
    step_count = obs_data.get("step_count", 0)
    feedback = obs_data.get("feedback", "")
    
    prompt = f"Task Description:\n{task_desc}\n\nSchema Info:\n{schema_info}\n\nSample Data:\n{sample_data}\n"
    if hints:
        prompt += f"\nHints: {', '.join(hints)}\n"
    if last_sql:
        prompt += f"\nLast SQL Submitted: {last_sql}\n"
    if last_result:
        prompt += f"Result of Last SQL: {last_result}\n"
    if last_error:
        prompt += f"Error Message: {last_error}\n"
    if step_count > 0 and feedback:
        prompt += f"Feedback from Grader: {feedback}\n"
        
    prompt += "\nRespond with JSON action only."
    return prompt

def parse_action(response_text: str, task_type: str) -> dict:
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        sql_match = re.search(r'(?i)SELECT\s+.*', response_text, re.DOTALL)
        sql = sql_match.group(0).strip() if sql_match else "SELECT 1"
        if sql.endswith('```'): sql = sql[:-3].strip()
        
        return {"action_type": task_type, "sql": sql, "explanation": "Fallback parsed"}

def run_episode(task_id: str) -> dict:
    res = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id})
    if res.status_code != 200:
        return {"task_id": task_id, "best_reward": 0.05}
    obs_obj = res.json()
    
    best_reward = 0.05
    for step in range(MAX_STEPS):
        obs = obs_obj.get("observation", obs_obj)
        if obs.get("done", False):
            break
            
        prompt = build_prompt(obs_obj)
        
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
            response_text = completion.choices[0].message.content or "{}"
        except Exception:
            response_text = "{}"
            
        action = parse_action(response_text, obs.get("task_type", "write_query"))
        
        res = requests.post(f"{ENV_URL}/step", json=action)
        if res.status_code != 200:
            break
        obs_obj = res.json()
        reward = obs_obj.get("reward", 0.05)
        
        best_reward = max(best_reward, reward)
        print(f"Step {step+1}: SQL={action.get('sql', '')[:60]}... Reward={reward:.3f} Feedback={obs_obj.get('observation', {}).get('feedback', '')}")
        
        if obs_obj.get("done", obs_obj.get("observation", {}).get("done", False)):
            break
            
    return {"task_id": task_id, "best_reward": round(best_reward, 3)}

def main():
    print(f"--- SQL Agent Inference ---")
    print(f"Model: {MODEL_NAME}")
    print(f"Env URL: {ENV_URL}")
    
    health = {"status": "unreachable"}
    try:
        health = requests.get(f"{ENV_URL}/health").json()
    except Exception:
        pass
    print(f"Health: {health}")
    
    task_ids = ["easy_01", "easy_02", "medium_01", "medium_02", "hard_01", "hard_02"]
    results = []
    
    for tid in task_ids:
        print(f"\n=== Task: {tid} ===")
        try:
            res = run_episode(tid)
        except Exception as e:
            res = {"task_id": tid, "best_reward": 0.05}
            print(f"Error running task {tid}: {e}")
        results.append(res)
        
    print("\n=== FINAL SCORES ===")
    easy_scores = []
    medium_scores = []
    hard_scores = []
    
    for r in results:
        t = r["task_id"]
        v = r["best_reward"]
        print(f"{t}: {v}")
        if t.startswith("easy"): easy_scores.append(v)
        elif t.startswith("medium"): medium_scores.append(v)
        elif t.startswith("hard"): hard_scores.append(v)
        
    easy_avg = sum(easy_scores)/len(easy_scores) if easy_scores else 0.0
    medium_avg = sum(medium_scores)/len(medium_scores) if medium_scores else 0.0
    hard_avg = sum(hard_scores)/len(hard_scores) if hard_scores else 0.0
    overall = (easy_avg + medium_avg + hard_avg) / 3.0
    
    print("\n--- AVERAGES ---")
    print(f"Easy Average:   {easy_avg:.3f}")
    print(f"Medium Average: {medium_avg:.3f}")
    print(f"Hard Average:   {hard_avg:.3f}")
    print(f"Overall Score:  {overall:.3f}")

if __name__ == "__main__":
    main()
