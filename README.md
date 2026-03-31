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

## The Goal
The objective is straightforward: the AI must interpret an internal database schema and translate human business instructions into valid SQL queries natively executed against the pipeline. 

Tasks increase in difficulty, pushing models to understand complex relationships such as:
- Simple Table Retrievals (`SELECT`, `WHERE`)
- Aggregating related tables (`JOIN`, `GROUP BY`)
- Complex constraint execution bounds (Logic matching)

## Fast Execution
1. **Install:** `pip install -r requirements.txt`
2. **Setup your preferred engine:**
   ```powershell
   $env:GEMINI_API_KEY="your-api-key" 
   # Or use OPENAI_API_KEY 
   ```
3. **Run the Simulation Base:**
   ```powershell
   python inference.py
   ```

## Architecture Workflow

This environment follows strict [OpenEnv Gym interface](https://github.com/meta-pytorch/OpenEnv) standards, utilizing `step()`, `reset()`, and `state()` endpoints wrapped locally or exposed natively via Hugging Face.

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
- `core/grader.py`: Evaluates logic correctness independent of string semantics by leveraging Pandas DataFrame alignments.
