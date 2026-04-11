import sqlite3
import re
from typing import Dict, Any, List, Optional
import json

SCHEMA_SQL = """
CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, department TEXT, salary INTEGER, hire_date TEXT, manager_id INTEGER);
CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT, budget INTEGER, location TEXT);
CREATE TABLE projects (id INTEGER PRIMARY KEY, name TEXT, department_id INTEGER, start_date TEXT, status TEXT);
CREATE TABLE project_assignments (employee_id INTEGER, project_id INTEGER, hours_worked INTEGER);
"""

SEED_DATA_SQL = """
INSERT INTO employees VALUES (1, 'Alice', 'Engineering', 95000, '2020-01-15', 3);
INSERT INTO employees VALUES (2, 'Bob', 'Engineering', 85000, '2021-02-20', 3);
INSERT INTO employees VALUES (3, 'Charlie', 'Engineering', 120000, '2019-01-10', NULL);
INSERT INTO employees VALUES (4, 'David', 'Sales', 70000, '2022-03-05', 5);
INSERT INTO employees VALUES (5, 'Eve', 'Sales', 92000, '2018-11-20', NULL);
INSERT INTO employees VALUES (6, 'Frank', 'HR', 65000, '2023-05-12', NULL);

INSERT INTO departments VALUES (1, 'Engineering', 500000, 'Building A');
INSERT INTO departments VALUES (2, 'Sales', 300000, 'Building B');
INSERT INTO departments VALUES (3, 'HR', 150000, 'Building C');

INSERT INTO projects VALUES (1, 'Project Alpha', 1, '2023-01-01', 'Active');
INSERT INTO projects VALUES (2, 'Project Beta', 2, '2023-06-15', 'Completed');

INSERT INTO project_assignments VALUES (1, 1, 100);
INSERT INTO project_assignments VALUES (2, 1, 150);
INSERT INTO project_assignments VALUES (4, 2, 80);
"""

TASKS = [
    {
        "id": "easy_01",
        "name": "Filter high earners",
        "difficulty": "easy",
        "type": "write_query",
        "grader": True,
        "description": "Find Engineering employees with salary > 90000, return name and salary ordered by salary DESC.",
        "expected_rows": 2
    },
    {
        "id": "easy_02",
        "name": "Count per department",
        "difficulty": "easy",
        "type": "write_query",
        "grader": True,
        "description": "Count employees per department, return department and count ordered by count DESC.",
        "expected_rows": 3
    },
    {
        "id": "medium_01",
        "name": "Fix broken JOIN",
        "difficulty": "medium",
        "type": "fix_query",
        "grader": True,
        "description": "Fix a broken JOIN query (missing ON keyword, wrong table alias in WHERE). Return name and department name.",
        "expected_rows": 6
    },
    {
        "id": "medium_02",
        "name": "Fix GROUP BY",
        "difficulty": "medium",
        "type": "fix_query",
        "grader": True,
        "description": "Fix wrong GROUP BY (should GROUP BY department not id) finding max salary per department.",
        "expected_rows": 3
    },
    {
        "id": "hard_01",
        "name": "Optimize correlated subquery",
        "difficulty": "hard",
        "type": "optimize_query",
        "grader": True,
        "description": "Replace correlated subquery with window function/CTE to find top earner per department with total hours.",
        "expected_rows": 3
    },
    {
        "id": "hard_02",
        "name": "Eliminate N+1 subqueries",
        "difficulty": "hard",
        "type": "optimize_query",
        "grader": True,
        "description": "Eliminate N+1 subqueries using LEFT JOIN + GROUP BY to compute total hours per project.",
        "expected_rows": 2
    }
]

def build_db() -> sqlite3.Connection:
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_SQL)
    conn.executescript(SEED_DATA_SQL)
    conn.commit()
    return conn

class SqlEnvironment:
    def __init__(self):
        self.db = build_db()
        self.max_steps = 8
        self.step_count = 0
        self.current_task = TASKS[0]
        self.done = False
        self.last_sql = ""
        self.last_result = ""
        self.last_error = ""
        self.reward = 0.05
        self.feedback = ""

    def _clamp(self, score: float) -> float:
        return round(max(0.05, min(0.95, score)), 4)

    def _execute_sql(self, sql: str) -> tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
        upper_sql = sql.upper()
        blocked = ["DROP", "DELETE", "INSERT", "UPDATE", "CREATE", "ALTER", "TRUNCATE"]
        for b in blocked:
            if re.search(rf"\b{b}\b", upper_sql):
                return None, f"Blocked keyword {b} found. Only SELECT allowed."
        try:
            cursor = self.db.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            return [dict(row) for row in rows], None
        except Exception as e:
            return None, str(e)

    def _grade_write(self, sql: str, rows: Optional[List[Dict]], expected_rows: int) -> float:
        score = 0.0
        upper_sql = sql.upper()
        if "EMPLOYEES" in upper_sql or "DEPARTMENTS" in upper_sql:
            score += 0.15
        if "WHERE" in upper_sql:
            score += 0.10
        if "ORDER BY" in upper_sql:
            score += 0.15
        if rows is not None:
            if len(rows) > 0:
                score += 0.25
            if len(rows) == expected_rows:
                score += 0.25
            elif abs(len(rows) - expected_rows) == 1:
                score += 0.12
        return self._clamp(score)

    def _grade_fix(self, sql: str, rows: Optional[List[Dict]], expected_rows: int) -> float:
        score = 0.0
        upper_sql = sql.upper()
        if rows is not None:
            score += 0.35
            if len(rows) == expected_rows:
                score += 0.35
        if re.search(r"\bJOIN\b.*\bON\b", upper_sql):
            score += 0.15
        if "WHERE" in upper_sql or "GROUP BY" in upper_sql:
            score += 0.10
        return self._clamp(score)

    def _grade_optimize(self, sql: str, rows: Optional[List[Dict]], expected_rows: int) -> float:
        score = 0.0
        upper_sql = sql.upper()
        if re.search(r"\b(WITH|ROW_NUMBER|RANK)\b", upper_sql):
            score += 0.30
        if "JOIN" in upper_sql:
            score += 0.25
        if upper_sql.count("SELECT") == 1:
            score += 0.20
        if rows is not None and len(rows) == expected_rows:
            score += 0.20
        return self._clamp(score)

    def reset(self, task_id: Optional[str] = None, difficulty: Optional[str] = None) -> Dict[str, Any]:
        self.step_count = 0
        self.done = False
        self.last_sql = ""
        self.last_result = ""
        self.last_error = ""
        self.reward = self._clamp(0.0)
        self.feedback = "Environment reset."
        
        if task_id:
            for t in TASKS:
                if t["id"] == task_id:
                    self.current_task = t
                    break
        elif difficulty:
            for t in TASKS:
                if t["difficulty"] == difficulty:
                    self.current_task = t
                    break
        else:
            self.current_task = TASKS[0]
            
        return self.get_observation()

    def get_observation(self) -> Dict[str, Any]:
        return {
            "task_id": self.current_task["id"],
            "task_type": self.current_task["type"],
            "task_description": self.current_task["description"],
            "schema_info": SCHEMA_SQL.strip(),
            "last_sql": self.last_sql,
            "last_result": self.last_result,
            "last_error": self.last_error,
            "step_count": self.step_count,
            "done": self.done,
            "reward": self.reward,
            "feedback": self.feedback
        }

    @property
    def state(self) -> Dict[str, Any]:
        return {
            "task": self.current_task["id"],
            "step": self.step_count,
            "done": self.done,
            "reward": self.reward
        }

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        if self.done:
            return self.get_observation()
            
        self.step_count += 1
        sql = action.get("sql", "").strip()
        self.last_sql = sql
        
        if not sql:
            self.last_error = "No SQL provided."
            self.last_result = ""
            self.reward = self._clamp(0.0)
            self.feedback = "Empty query."
        else:
            rows, err = self._execute_sql(sql)
            if err:
                self.last_error = err
                self.last_result = ""
                self.reward = self._clamp(0.0)
                self.feedback = "Execution failed."
            else:
                self.last_error = ""
                self.last_result = json.dumps(rows)
                expected = self.current_task["expected_rows"]
                ttype = self.current_task["type"]
                
                raw_score = 0.0
                if ttype == "write_query":
                    raw_score = self._grade_write(sql, rows, expected)
                elif ttype == "fix_query":
                    raw_score = self._grade_fix(sql, rows, expected)
                elif ttype == "optimize_query":
                    raw_score = self._grade_optimize(sql, rows, expected)
                
                self.reward = raw_score # already clamped internally by the functions
                self.feedback = "Query executed successfully."

        if self.step_count >= self.max_steps:
            self.done = True
            
        return self.get_observation()
