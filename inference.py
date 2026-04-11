"""
SQL Data Analyst OpenEnv — baseline inference (hackathon format).

Environment variables (set before running):
  HF_TOKEN         Primary API key (Hugging Face / OpenAI-compatible).
  API_BASE_URL     LLM base URL (e.g. https://api.openai.com/v1 or HF router).
  MODEL_NAME       Model id for chat completions.
  OPENAI_API_KEY   Optional fallback if HF_TOKEN is unset.
  API_KEY          Optional second fallback.

Optional:
  SQL_AGENT_TASK      Logged as task= in [START] (default: sql_analyst_episode).
  SQL_AGENT_BENCHMARK Logged as env= in [START] (default: sql_agent_openenv).
  MAX_STEPS           Max env.step calls (default: 24).
  OPENAI_SEED         Passed to OpenAI only when base URL looks like OpenAI.
  ENV_CONTAINER_START true to use SqlEnvClient.from_docker_image (default: false).
  IMAGE_NAME / LOCAL_IMAGE_NAME / ENV_IMAGE_NAME  Docker image tag when using container.
"""

from __future__ import annotations

import os
import re
import sys
import textwrap
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from client import SqlEnvClient
from models import SqlAction

# --- Mandatory hackathon configuration ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_KEY = (
    os.getenv("HF_TOKEN")
    or os.getenv("OPENAI_API_KEY")
    or os.getenv("API_KEY")
)

TASK_NAME = os.getenv("SQL_AGENT_TASK", "sql_analyst_episode")
BENCHMARK = os.getenv("SQL_AGENT_BENCHMARK", "sql_agent_openenv")

MAX_STEPS = int(os.getenv("MAX_STEPS", "24"))
OPENAI_SEED = int(os.getenv("OPENAI_SEED", "42"))

ENV_CONTAINER_START = os.getenv("ENV_CONTAINER_START", "false").lower() == "true"
ENV_IMAGE_NAME = (
    os.getenv("LOCAL_IMAGE_NAME")
    or os.getenv("IMAGE_NAME")
    or os.getenv("ENV_IMAGE_NAME", "sql-agent-env:latest")
)

# Grader task ids (must match core/tasks.py) for normalized [END] score in [0, 1]
TASK_IDS = [
    "employee_payroll_overview",
    "department_budget_summary",
    "senior_engineering_comp_review",
]

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert Data Analyst and SQL Agent.
    You will be provided with a SQL Schema and an instruction.
    Reply with exactly one SQL query string to execute.
    Do NOT use markdown fences or explanations — only the raw SQL text.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}")

def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    err_one = sanitize_one_line(error) if error else "null"
    act_one = sanitize_one_line(action)
    done_val = "true" if done else "false"
    print(f"[STEP]  step={step} action={act_one} reward={reward:.2f} done={done_val} error={err_one}")

def log_end(success: bool, steps: int, score: float) -> None:
    succ_val = "true" if success else "false"
    print(f"[END]   success={succ_val} steps={steps} rewards={score:.2f}")


def sanitize_one_line(s: str) -> str:
    if not s:
        return ""
    return " ".join(s.split())


def build_user_prompt(step: int, observation: Any, history: List[str]) -> str:
    schema = observation.schema_info
    instruction = observation.current_task_instruction
    exec_result = observation.execution_result or "None"
    error = observation.execution_error or "None"
    history_block = "\n".join(history[-6:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Database Schema:
        {schema}

        Task Instruction:
        {instruction}

        Previous Execution Result: {exec_result}
        Previous Error: {error}

        Recent history:
        {history_block}

        Write the precise SQL query string to accomplish the task instruction. Return nothing else.
        """
    ).strip()


def parse_model_action(response_text: str) -> str:
    query = (response_text or "").strip()
    if query.startswith("```sql"):
        query = query[6:]
    if query.startswith("```"):
        query = query[3:]
    if query.endswith("```"):
        query = query[:-3]
    return query.strip()


def _make_direct_client():
    from server.environment import SqlEnvironment

    base_env = SqlEnvironment()

    class DirectClient:
        def __init__(self, target):
            self.target = target

        def reset(self):
            obs = self.target.reset()
            return type(
                "StepResult",
                (),
                {"observation": obs, "reward": 0.0, "done": False},
            )()

        def step(self, action):
            obs = self.target.step(action)
            return type(
                "StepResult",
                (),
                {
                    "observation": obs,
                    "reward": obs.reward if obs.reward is not None else 0.0,
                    "done": obs.done,
                },
            )()

        def close(self):
            pass

    return DirectClient(base_env)


def main() -> None:
    if not API_KEY:
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    if ENV_CONTAINER_START:
        env = SqlEnvClient.from_docker_image(ENV_IMAGE_NAME).sync()
    else:
        env = _make_direct_client()

    history: List[str] = []
    
    try:
        result = env.reset()
        observation = result.observation
        
        current_task_id = observation.task_id
        task_step = 1
        log_start(task=current_task_id, env=BENCHMARK, model=MODEL_NAME)

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            user_prompt = build_user_prompt(step, observation, history)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]

            create_kwargs: Dict[str, Any] = {
                "model": MODEL_NAME,
                "messages": messages,
                "temperature": 0.0,
                "stream": False,
            }
            if re.search(r"openai\.com", API_BASE_URL, re.I):
                create_kwargs["seed"] = OPENAI_SEED

            try:
                completion = client.chat.completions.create(**create_kwargs)
                response_text = completion.choices[0].message.content or "SELECT 1"
            except Exception:
                response_text = "SELECT 1"

            action_str = parse_model_action(response_text)
            result = env.step(SqlAction(query=action_str))
            observation = result.observation
            
            # reward is obtained from observation task_score or directly from step
            task_score = float(observation.task_score if hasattr(observation, 'task_score') and observation.task_score is not None else 0.0)
            if hasattr(observation, 'reward') and observation.reward is not None:
                task_score = float(observation.reward)
                
            err_raw = observation.execution_error
            next_task_id = getattr(observation, "task_id", None)
            
            # task transitions or episode done signals end of current task
            task_done = (next_task_id != current_task_id) or result.done
            
            log_step(step=task_step, action=action_str, reward=task_score, done=task_done, error=err_raw)

            history.append(f"Q: {action_str[:200]} -> R: {task_score:+.2f}")

            if task_done:
                success = (task_score >= 0.95)
                log_end(success=success, steps=task_step, score=task_score)
                if result.done or not next_task_id:
                    break
                current_task_id = next_task_id
                task_step = 1
                log_start(task=current_task_id, env=BENCHMARK, model=MODEL_NAME)
                history.clear()
            else:
                task_step += 1

    except Exception:
        pass
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
