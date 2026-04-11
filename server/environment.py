import sqlite3
import re
from typing import Dict, Any, List, Optional
import json

SCHEMA_SQL = """
CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, department TEXT, salary REAL, hire_date TEXT, manager_id INTEGER);
CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT, budget REAL, location TEXT);
CREATE TABLE projects (id INTEGER PRIMARY KEY, name TEXT, department_id INTEGER, start_date TEXT, status TEXT);
CREATE TABLE project_assignments (employee_id INTEGER, project_id INTEGER, hours_worked REAL, PRIMARY KEY(employee_id, project_id));
"""

SEED_DATA_SQL = """
INSERT INTO departments VALUES (1,'Engineering',500000,'New York'), (2,'Marketing',200000,'Chicago'), (3,'Sales',300000,'Los Angeles');
INSERT INTO employees VALUES (1,'Alice Chen','Engineering',95000,'2020-01-15',NULL), (2,'Bob Smith','Engineering',85000,'2021-03-20',1), (3,'Carol Davis','Marketing',72000,'2019-07-01',NULL), (4,'David Lee','Sales',68000,'2022-11-01',NULL), (5,'Eve Turner','Engineering',110000,'2018-05-10',1), (6,'Frank White','Marketing',65000,'2023-01-15',3);
INSERT INTO projects VALUES (1,'Data Platform',1,'2023-01-01','active'), (2,'Website Redesign',2,'2023-03-15','completed'), (3,'CRM Migration',3,'2023-06-01','active');
INSERT INTO project_assignments VALUES (1,1,120),(2,1,80),(5,1,200), (3,2,160),(6,2,40),(4,3,100);
"""

TASKS = {
  "easy_01": {
    "type": "write_query",
    "difficulty": "easy",
    "description": "Find all employees in the Engineering department with salary above 90000. Return name and salary ordered by salary descending.",
    "expected_columns": ["name", "salary"],
    "expected_row_count": 2,
    "hints": ["Filter department = Engineering", "salary > 90000", "ORDER BY salary DESC"]
  },
  "easy_02": {
    "type": "write_query", 
    "difficulty": "easy",
    "description": "Count how many employees are in each department. Return department and employee count ordered by count descending.",
    "expected_columns": ["department", "count"],
    "expected_row_count": 3,
    "hints": ["Use GROUP BY department", "Use COUNT(*)"]
  },
  "medium_01": {
    "type": "fix_query",
    "difficulty": "medium",
    "description": "Fix this broken query that returns employees on active projects with project name and hours worked.",
    "broken_sql": "SELECT e.name, p.name, pa.hours_worked FROM employees e JOIN project_assignments pa e.id = pa.employee_id JOIN projects p ON pa.project_id = p.id WHERE projects.status = 'active'",
    "expected_row_count": 3,
    "bugs": ["Missing ON keyword in first JOIN", "Wrong alias 'projects' should be 'p' in WHERE"]
  },
  "medium_02": {
    "type": "fix_query",
    "difficulty": "medium", 
    "description": "Fix this query that should return average salary per department but groups incorrectly.",
    "broken_sql": "SELECT department, AVG(salary) FROM employees GROUP BY id ORDER BY AVG(salary) DESC",
    "expected_row_count": 3,
    "bugs": ["GROUP BY should use department not id"]
  },
  "hard_01": {
    "type": "optimize_query",
    "difficulty": "hard",
    "description": "Optimize this slow query: find top earner in each department with their total hours worked. Replace correlated subqueries with window functions or CTEs.",
    "slow_sql": "SELECT e.name, e.department, e.salary, (SELECT SUM(hours_worked) FROM project_assignments pa WHERE pa.employee_id = e.id) as total_hours FROM employees e WHERE e.salary = (SELECT MAX(salary) FROM employees e2 WHERE e2.department = e.department) ORDER BY e.salary DESC",
    "expected_row_count": 3,
    "good_patterns": ["ROW_NUMBER", "RANK", "WITH ", "LEFT JOIN"],
    "optimization_hints": ["Use window functions", "Use CTEs"]
  },
  "hard_02": {
    "type": "optimize_query",
    "difficulty": "hard",
    "description": "Rewrite this N+1 query: return each department name, budget, employee count, and active project count using JOINs instead of subqueries.",
    "slow_sql": "SELECT d.name, d.budget, (SELECT COUNT(*) FROM employees e WHERE e.department = d.name) as emp_count, (SELECT COUNT(*) FROM projects p WHERE p.department_id = d.id AND p.status='active') as active_projects FROM departments d",
    "expected_row_count": 3,
    "good_patterns": ["LEFT JOIN", "GROUP BY", "COUNT("],
    "optimization_hints": ["Use LEFT JOIN with GROUP BY"]
  }
}

class SQLEnvironment:
    def __init__(self):
        self.step_count = 0
        self.done = False
        self.reward = 0.05
        self.feedback = ""
        self.last_sql = ""
        self.last_result = ""
        self.last_error = ""
        self.task_id = "easy_01"
        self.task = TASKS[self.task_id]
        self.conn = None
        self._build_db()

    def _build_db(self):
        if self.conn:
            self.conn.close()
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA_SQL)
        self.conn.executescript(SEED_DATA_SQL)
        self.conn.commit()
        return self.conn

    def _clamp(self, score: float) -> float:
        return round(max(0.05, min(0.95, score)), 4)

    def reset(self, task_id=None, difficulty=None):
        self.step_count = 0
        self.done = False
        self.last_sql = ""
        self.last_result = ""
        self.last_error = ""
        self.reward = self._clamp(0.15)
        self.feedback = "Environment reset."

        if task_id and task_id in TASKS:
            self.task_id = task_id
            self.task = TASKS[task_id]
        elif difficulty:
            candidates = [k for k, v in TASKS.items() if v["difficulty"] == difficulty]
            if candidates:
                self.task_id = candidates[0]
                self.task = TASKS[self.task_id]
        else:
            self.task_id = list(TASKS.keys())[0]
            self.task = TASKS[self.task_id]

        self.conn = self._build_db()

        if self.task["type"] == "fix_query":
            self.last_sql = self.task["broken_sql"]
        elif self.task["type"] == "optimize_query":
            self.last_sql = self.task["slow_sql"]

        return self._build_observation(self.reward, self.feedback)

    def _execute_sql(self, sql: str):
        upper_sql = sql.upper()
        blocked = ["DROP", "DELETE", "INSERT", "UPDATE", "CREATE", "ALTER", "TRUNCATE"]
        for b in blocked:
            if re.search(rf"\b{b}\b", upper_sql):
                return None, "Forbidden operation"
                
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            if not rows:
                return json.dumps([{}]) if cursor.description else "[]", None
            
            res_dict = [dict(r) for r in rows[:10]]
            return json.dumps(res_dict), None
        except Exception as e:
            return None, str(e)

    def _grade_write(self, sql, rows, cols) -> tuple[float, str]:
        score = 0.0
        feedback_parts = []
        lower_sql = sql.lower()
        
        if "employees" in lower_sql or "departments" in lower_sql:
            score += 0.15
            feedback_parts.append("Correct table used")
            
        if "where" in lower_sql:
            score += 0.10
            feedback_parts.append("WHERE clause present")
            
        expected_cols = [c.lower() for c in self.task["expected_columns"]]
        actual_cols = [c.lower() for c in cols]
        matched = sum(1 for ec in expected_cols if ec in actual_cols)
        if expected_cols:
            score += (matched / len(expected_cols)) * 0.25
            feedback_parts.append(f"Columns: {matched}/{len(expected_cols)} matched")
            
        expected_rows = self.task["expected_row_count"]
        if rows is not None:
            if len(rows) == expected_rows:
                score += 0.25
                feedback_parts.append(f"Row count correct: {len(rows)}")
            elif abs(len(rows) - expected_rows) == 1:
                score += 0.12
                feedback_parts.append(f"Row count close: {len(rows)} vs {expected_rows}")
            else:
                feedback_parts.append(f"Row count wrong: {len(rows)} vs {expected_rows}")
                
        if "order by" in lower_sql:
            score += 0.15
            feedback_parts.append("ORDER BY present")
            
        return self._clamp(score), " | ".join(feedback_parts)

    def _grade_fix(self, sql, rows, error) -> tuple[float, str]:
        score = 0.0
        feedback_parts = []
        lower_sql = sql.lower()
        
        if error is None and rows is not None:
            score += 0.35
            feedback_parts.append("Query executes successfully")
        else:
            feedback_parts.append(f"Still broken: {error}")
            
        expected_rows = self.task["expected_row_count"]
        if rows is not None and len(rows) == expected_rows:
            score += 0.35
            feedback_parts.append(f"Correct rows: {len(rows)}")
            
        if "join" in lower_sql and " on " in lower_sql:
            score += 0.15
            feedback_parts.append("JOIN...ON syntax correct")
            
        if "group by" in lower_sql or "where" in lower_sql:
            score += 0.10
            feedback_parts.append("Filter/grouping present")
            
        return self._clamp(score), " | ".join(feedback_parts)

    def _grade_optimize(self, sql, rows) -> tuple[float, str]:
        score = 0.0
        feedback_parts = []
        sql_upper = sql.upper()
        
        if "WITH " in sql_upper or "ROW_NUMBER" in sql_upper or "RANK" in sql_upper:
            score += 0.30
            feedback_parts.append("Uses CTE or window function")
            
        if "JOIN" in sql_upper:
            score += 0.25
            feedback_parts.append("Uses JOIN")
            
        select_count = sql_upper.count("SELECT")
        if select_count == 1:
            score += 0.20
            feedback_parts.append("No nested SELECT (no correlated subqueries)")
        else:
            feedback_parts.append(f"Still has {select_count - 1} nested SELECT(s)")
            
        expected_rows = self.task["expected_row_count"]
        if rows is not None and len(rows) == expected_rows:
            score += 0.20
            feedback_parts.append(f"Correct rows: {len(rows)}")
            
        return self._clamp(score), " | ".join(feedback_parts)

    def get_schema_info(self) -> str:
        return SCHEMA_SQL.strip()
        
    def get_sample_data(self) -> str:
        samples = []
        tables = ["departments", "employees", "projects", "project_assignments"]
        cursor = self.conn.cursor()
        for t in tables:
            cursor.execute(f"SELECT * FROM {t} LIMIT 3")
            rows = cursor.fetchall()
            cols = [desc[0] for desc in cursor.description] if cursor.description else []
            samples.append(f"Table {t}:\nCols: {', '.join(cols)}")
            for r in rows:
                samples.append(str(dict(r)))
        return "\n".join(samples)

    def _build_observation(self, reward, feedback, last_sql=None, last_result=None, last_error=None):
        reward = self._clamp(reward)
        obs = {
            "task_id": self.task_id,
            "task_type": self.task["type"],
            "task_description": self.task["description"],
            "schema_info": self.get_schema_info(),
            "sample_data": self.get_sample_data(),
            "last_sql": last_sql or self.last_sql,
            "last_result": last_result or self.last_result,
            "last_error": last_error or self.last_error,
            "step_count": self.step_count,
            "done": self.done,
            "reward": reward,
            "feedback": feedback,
            "metadata": self.task.get("hints", []) + self.task.get("good_patterns", [])
        }
        return {"observation": obs, "reward": reward, "done": self.done, "info": {}}

    @property
    def state(self):
        return {
            "episode_id": "ep_" + str(self.step_count),
            "task_type": self.task["type"],
            "task_id": self.task_id,
            "step_count": self.step_count,
            "total_reward": self.reward,
            "best_score_so_far": self.reward,
            "attempts": self.step_count
        }

    def step(self, action: dict):
        self.step_count += 1
        sql = action.get("sql", "").strip()
        self.last_sql = sql
        
        if not sql:
            self.reward = self._clamp(0.0)
            self.last_error = "No SQL provided"
            self.last_result = ""
            self.feedback = "Missing SQL"
        else:
            res_text, err = self._execute_sql(sql)
            if err:
                self.last_error = err
                self.last_result = ""
                self.reward = self._clamp(0.08)
                self.feedback = err
            else:
                self.last_error = ""
                self.last_result = res_text
                
                rows = []
                cols = []
                if res_text and res_text != "[]" and res_text != "[{}]":
                    try:
                        rows = json.loads(res_text)
                        if rows and isinstance(rows[0], dict):
                            cols = list(rows[0].keys())
                    except:
                        pass
                
                ttype = self.task["type"]
                if ttype == "write_query":
                    self.reward, self.feedback = self._grade_write(sql, rows, cols)
                elif ttype == "fix_query":
                    self.reward, self.feedback = self._grade_fix(sql, rows, err)
                elif ttype == "optimize_query":
                    self.reward, self.feedback = self._grade_optimize(sql, rows)

        if self.reward > 0.90 or self.step_count >= 8:
            self.done = True

        return self._build_observation(self.reward, self.feedback)
