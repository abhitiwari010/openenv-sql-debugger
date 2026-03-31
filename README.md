---
title: SQL AI Data Analyst
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
---
# OpenEnv SQL Data Analyst

## Environment Description and Motivation
This environment simulates a real business workflow: a data analyst converting stakeholder requests into executable SQL over an internal company warehouse.

Why this is real-world:
- HR and finance teams routinely need ad-hoc SQL analysis.
- Models must handle schema understanding, joins, aggregation, and constraints.
- Evaluation is done against deterministic graders over actual query results (not string matching).

The environment implements the OpenEnv lifecycle (`reset()`, `step()`, `state()`) and is deployable on Hugging Face Spaces as a containerized service.

## Action Space
`SqlAction`
- `query: str` - a SQL query to execute against the provided SQLite schema.

## Observation Space
`SqlObservation`
- `current_task_instruction: str` - current objective text.
- `schema_info: str` - schema description.
- `task_id: str` - stable task identifier.
- `difficulty: str` - `easy | medium | hard`.
- `execution_result: Optional[str]` - previous query result rows as JSON.
- `execution_error: Optional[str]` - previous execution or safety error.
- `task_score: float` - grader output in `[0.0, 1.0]`.
- `grader_feedback: Optional[str]` - deterministic grader feedback.

## State
`SqlState`
- Tracks episode id, step count, current task index, accumulated reward, per-task scores, and attempts.
- Exposed via `state()` for inspection and reproducibility.

## Tasks and Difficulty Progression
1. **Easy - `employee_payroll_overview`**  
   List employee names and salaries in descending salary order.
2. **Medium - `department_budget_summary`**  
   Compute average salary per department with required output schema.
3. **Hard - `senior_engineering_comp_review`**  
   Multi-table logic to identify engineering employees meeting senior salary thresholds.

All tasks have deterministic graders producing scores from `0.0` to `1.0`.

## Reward Function
Reward is shaped for trajectory-level signal (not binary terminal only):
- `+0.1` valid SQL execution bonus
- `+0.9 * task_score` progress toward correctness
- `-0.02` per step (discourages loops)
- `-0.5` invalid query penalty
- `-1.0` safety penalty for destructive SQL (`DROP/DELETE/TRUNCATE/ALTER/UPDATE/INSERT`)

Task completion threshold is `task_score >= 0.95`.
Episode ends when all tasks are solved or max attempts for a task are exhausted.

## Setup and Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set API key for baseline:
   ```bash
   export OPENAI_API_KEY="your-key"
   ```
3. Run baseline:
   ```bash
   python inference.py
   ```

The baseline uses the OpenAI API client with deterministic settings:
- `temperature=0.0`
- fixed seed (`OPENAI_SEED`, default `42`)
- fixed task order and deterministic grader

## Baseline Score Reporting
`inference.py` prints:
- per-task best score
- average task score
- episode return
- model and seed used

This creates reproducible benchmark records for easy/medium/hard tasks.

## Docker and Hugging Face Space
- `Dockerfile` is included and starts `uvicorn server.app:app`.
- Space metadata is configured in this `README.md` frontmatter with `sdk: docker`.
- Health endpoint: `/health`

Local container run:
```bash
docker build -t sql-agent-env .
docker run -p 7860:7860 sql-agent-env
```

## OpenEnv Metadata
OpenEnv runtime metadata is declared in `openenv.yaml`.

Validation command:
```bash
openenv validate
```

```mermaid
flowchart TD
    Start([Episode Start]) --> EnvInit[Environment loads Task 1]
    
    EnvInit --> Obs[Generate Observation]
    Obs -.-> |1. Send Schema and Instruction| Agent((LLM Agent))
    
    Agent -.-> |2. Predicts SQL Query| Step[Environment step function]
    
    Step --> Exec{Execute SQL on SQLite Engine}
    
    Exec -->|SQL Syntax Error| Pen[Return Negative Reward]
    Exec -->|Valid SQL| Grader[Evaluate Pandas DataFrame]
    
    Grader --> Match{Compare Expected DataFrame}
    Match -->|Perfect Match| Rew1[Reward 1.0 Advance Level]
    Match -->|Unordered Match| RewPartial[Reward 0.8]
    Match -->|Incorrect Data| Rew0[Reward 0.0]
    
    Pen --> StateUpdate
    Rew1 --> StateUpdate
    RewPartial --> StateUpdate
    Rew0 --> StateUpdate
    
    StateUpdate[Log History and Update Episode state] --> CheckDone{All Tasks Completed Yes or No}
    
    CheckDone -->|No| Obs
    CheckDone -->|Yes| Finish([Episode End])
```

## Internal Engine Design
To read more about exactly how the logic is handled under the hood, access the OOP components inside the root directory:
- `server/environment.py`: The orchestrator handling the OpenEnv lifecycle.
- `database/sqlite_manager.py`: Creates ephemeral SQLite sandboxes per session.
- `core/grader.py`: Deterministic result grader with partial credit and detailed feedback.
