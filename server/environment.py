import sqlite3, uuid, random, re
from typing import Optional, Tuple, List, Dict, Any

SCHEMA_SQL = """
CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    department TEXT NOT NULL,
    salary REAL NOT NULL,
    hire_date TEXT NOT NULL,
    manager_id INTEGER REFERENCES employees(id)
);

CREATE TABLE departments (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    budget REAL NOT NULL,
    location TEXT NOT NULL
);

CREATE TABLE projects (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    department_id INTEGER REFERENCES departments(id),
    start_date TEXT NOT NULL,
    status TEXT NOT NULL
);

CREATE TABLE project_assignments (
    employee_id INTEGER REFERENCES employees(id),
    project_id INTEGER REFERENCES projects(id),
    hours_worked REAL NOT NULL,
    PRIMARY KEY (employee_id, project_id)
);
"""

SEED_SQL = """
INSERT INTO departments VALUES
(1,'Engineering',500000,'New York'),
(2,'Marketing',200000,'Chicago'),
(3,'Sales',300000,'Los Angeles');

INSERT INTO employees VALUES
(1,'Alice Chen','Engineering',95000,'2020-01-15',NULL),
(2,'Bob Smith','Engineering',85000,'2021-03-20',1),
(3,'Carol Davis','Marketing',72000,'2019-07-01',NULL),
(4,'David Lee','Sales',68000,'2022-11-01',NULL),
(5,'Eve Turner','Engineering',110000,'2018-05-10',1),
(6,'Frank White','Marketing',65000,'2023-01-15',3);

INSERT INTO projects VALUES
(1,'Data Platform',1,'2023-01-01','active'),
(2,'Website Redesign',2,'2023-03-15','completed'),
(3,'CRM Migration',3,'2023-06-01','active');

INSERT INTO project_assignments VALUES
(1,1,120),(2,1,80),(5,1,200),
(3,2,160),(6,2,40),(4,3,100);
"""

TASKS = {
  "easy_01": {
    "type": "write_query",
    "difficulty": "easy",
    "description": "Find all employees in the Engineering department with salary above 90000. Return their name and salary, ordered by salary descending.",
    "expected_columns": ["name", "salary"],
    "expected_row_count": 2,
    "expected_rows": [
      {"name": "Eve Turner", "salary": 110000.0},
      {"name": "Alice Chen", "salary": 95000.0}
    ],
    "hints": ["Filter WHERE department = 'Engineering'", "AND salary > 90000", "ORDER BY salary DESC"]
  },
  "easy_02": {
    "type": "write_query",
    "difficulty": "easy",
    "description": "Count how many employees are in each department. Return department name and employee count, ordered by count descending.",
    "expected_columns": ["department", "count"],
    "expected_row_count": 3,
    "hints": ["Use GROUP BY department", "Use COUNT(*)", "ORDER BY count DESC"]
  },
  "medium_01": {
    "type": "fix_query",
    "difficulty": "medium",
    "description": "Fix this broken query that should return all employees working on active projects along with the project name and hours worked.",
    "broken_sql": "SELECT e.name, p.name, pa.hours_worked\nFROM employees e\nJOIN project_assignments pa e.id = pa.employee_id\nJOIN projects p ON pa.project_id = p.id\nWHERE projects.status = 'active'\n",
    "expected_row_count": 3,
    "bugs": ["Missing ON keyword between 'pa' and 'e.id' in first JOIN", "Wrong table reference: 'projects.status' should be 'p.status' in WHERE"]
  },
  "medium_02": {
    "type": "fix_query",
    "difficulty": "medium",
    "description": "Fix this query that should return average salary per department but is producing wrong results because it groups by the wrong column.",
    "broken_sql": "SELECT department, AVG(salary)\nFROM employees\nGROUP BY id\nORDER BY AVG(salary) DESC\n",
    "expected_row_count": 3,
    "bugs": ["GROUP BY should use 'department' not 'id'"]
  },
  "hard_01": {
    "type": "optimize_query",
    "difficulty": "hard",
    "description": "This query works but is slow. Optimize it to find the top earner in each department along with their total hours worked across all projects. Replace correlated subqueries with window functions or CTEs.",
    "slow_sql": "SELECT e.name, e.department, e.salary,\n       (SELECT SUM(hours_worked) \n        FROM project_assignments pa \n        WHERE pa.employee_id = e.id) as total_hours\nFROM employees e\nWHERE e.salary = (SELECT MAX(salary) \n                  FROM employees e2 \n                  WHERE e2.department = e.department)\nORDER BY e.salary DESC\n",
    "expected_row_count": 3,
    "good_patterns": ["WITH ", "ROW_NUMBER", "RANK", "LEFT JOIN"],
    "bad_patterns": ["SELECT.*SELECT.*SELECT"],
    "optimization_hints": ["Use a CTE (WITH clause) to pre-aggregate hours", "Use ROW_NUMBER() or RANK() window function", "Replace nested SELECT with LEFT JOIN"]
  },
  "hard_02": {
    "type": "optimize_query",
    "difficulty": "hard",
    "description": "Rewrite this slow N+1 query to return each department name, total budget, number of employees, and number of active projects. Use JOINs instead of correlated subqueries.",
    "slow_sql": "SELECT d.name, d.budget,\n       (SELECT COUNT(*) FROM employees e \n        WHERE e.department = d.name) as emp_count,\n       (SELECT COUNT(*) FROM projects p \n        WHERE p.department_id = d.id \n        AND p.status='active') as active_projects\nFROM departments d\n",
    "expected_row_count": 3,
    "good_patterns": ["LEFT JOIN", "GROUP BY", "COUNT("],
    "optimization_hints": ["Use LEFT JOIN employees ON department name", "Use LEFT JOIN projects ON department_id", "Use GROUP BY d.id or d.name", "Use COUNT() in SELECT"]
  }
}

def build_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_SQL)
    conn.executescript(SEED_SQL)
    conn.commit()
    return conn

def get_schema_info() -> str:
    return SCHEMA_SQL.strip()

def get_sample_data() -> str:
    conn = build_db()
    lines = []
    for table in ["employees", "departments", "projects", "project_assignments"]:
        cursor = conn.execute(f"SELECT * FROM {table} LIMIT 3")
        cols = [d[0] for d in cursor.description]
        rows = cursor.fetchall()
        lines.append(f"\\n-- {table} (first 3 rows)")
        lines.append(", ".join(cols))
        for row in rows:
            lines.append(", ".join(str(v) for v in row))
    conn.close()
    return "\\n".join(lines)

class SQLEnvironment:
    def __init__(self):
        self.episode_id = ""
        self.step_count = 0
        self.total_reward = 0.0
        self.best_score = 0.0
        self.done = False
        self.task = None
        self.task_id = None
        self.conn = None
        self.attempts = 0

    def _clamp(self, score: float) -> float:
        return round(max(0.05, min(0.95, float(score))), 4)

    def _reset_state(self):
        self.episode_id = str(uuid.uuid4())
        self.step_count = 0
        self.total_reward = 0.0
        self.best_score = 0.0
        self.done = False
        self.attempts = 0
        if self.conn:
            try:
                self.conn.close()
            except:
                pass
        self.conn = build_db()

    def reset(self, task_id=None, difficulty=None) -> dict:
        self._reset_state()
        
        available = list(TASKS.keys())
        if task_id and task_id in TASKS:
            self.task_id = task_id
        elif difficulty:
            pool = [k for k,v in TASKS.items() if v["difficulty"] == difficulty]
            self.task_id = random.choice(pool) if pool else random.choice(available)
        else:
            self.task_id = random.choice(available)
            
        self.task = TASKS[self.task_id]
        
        starter_sql = (self.task.get("broken_sql") or self.task.get("slow_sql") or "")
        
        return self._build_result(
            reward=0.15,
            feedback="New episode started. Study the schema and task carefully.",
            last_sql=starter_sql.strip() if starter_sql else None,
            last_result=None,
            last_error=None
        )

    def step(self, action: dict) -> dict:
        if self.done:
            return self._build_result(
                reward=self._clamp(0.0),
                feedback="Episode complete. Call reset() to start a new episode."
            )
            
        self.step_count += 1
        self.attempts += 1
        sql = (action.get("sql") or "").strip()
        
        if not sql:
            return self._build_result(
                reward=self._clamp(0.0),
                feedback="Empty SQL submitted. Please write a SQL SELECT statement."
            )
            
        result_text, error_text = self._execute_sql(sql)
        
        if error_text:
            reward = self._clamp(0.04)
            self.total_reward += reward
            return self._build_result(
                reward=reward,
                feedback=f"SQL Error: {error_text}. Fix the syntax and try again.",
                last_sql=sql,
                last_result=None,
                last_error=error_text
            )
            
        task_type = self.task["type"]
        if task_type == "write_query":
            rows, cols = self._get_rows_cols(sql)
            reward, feedback = self._grade_write(sql, rows, cols)
        elif task_type == "fix_query":
            rows, cols = self._get_rows_cols(sql)
            reward, feedback = self._grade_fix(sql, rows, error_text)
        elif task_type == "optimize_query":
            rows, cols = self._get_rows_cols(sql)
            reward, feedback = self._grade_optimize(sql, rows)
        else:
            reward, feedback = self._clamp(0.0), "Unknown task type."
            
        self.total_reward += reward
        self.best_score = max(self.best_score, reward)
        
        if reward >= 0.90 or self.step_count >= 8:
            self.done = True
            
        return self._build_result(
            reward=reward,
            feedback=feedback,
            last_sql=sql,
            last_result=result_text,
            last_error=None
        )

    def _execute_sql(self, sql: str) -> Tuple[Optional[str], Optional[str]]:
        forbidden = ["DROP", "DELETE", "INSERT", "UPDATE", "CREATE", "ALTER", "TRUNCATE"]
        sql_upper = sql.upper()
        for kw in forbidden:
            if kw in sql_upper:
                return None, f"Forbidden: {kw} not allowed. Only SELECT."
                
        try:
            cursor = self.conn.execute(sql)
            if cursor.description is None:
                return "Query executed (no rows returned).", None
            cols = [d[0] for d in cursor.description]
            rows = cursor.fetchall()
            if not rows:
                return "Query ran but returned 0 rows.", None
            lines = [", ".join(cols)]
            for row in rows[:10]:
                lines.append(", ".join(str(v) for v in row))
            if len(rows) > 10:
                lines.append(f"... ({len(rows)} total rows, showing first 10)")
            return "\\n".join(lines), None
        except sqlite3.Error as e:
            return None, str(e)

    def _get_rows_cols(self, sql: str):
        forbidden = ["DROP","DELETE","INSERT","UPDATE","CREATE","ALTER","TRUNCATE"]
        if any(kw in sql.upper() for kw in forbidden):
            return [], []
        try:
            cursor = self.conn.execute(sql)
            if cursor.description is None:
                return [], []
            cols = [d[0] for d in cursor.description]
            rows = cursor.fetchall()
            return rows, cols
        except:
            return [], []

    def _grade_write(self, sql: str, rows: list, cols: list) -> Tuple[float, str]:
        score = 0.0
        parts = []
        sql_lower = sql.lower()
        
        task_tables = ["employees", "departments", "projects"]
        if any(t in sql_lower for t in task_tables):
            score += 0.15
            parts.append("Correct table used (+0.15)")
        else:
            parts.append("Wrong or missing table")
            
        if "where" in sql_lower:
            score += 0.10
            parts.append("WHERE clause present (+0.10)")
        else:
            parts.append("No WHERE clause")
            
        expected_cols = [c.lower() for c in self.task.get("expected_columns", [])]
        actual_cols = [c.lower() for c in cols]
        if expected_cols and actual_cols:
            matched = sum(1 for ec in expected_cols if any(ec in ac or ac in ec for ac in actual_cols))
            col_score = (matched / len(expected_cols)) * 0.25
            score += col_score
            parts.append(f"Columns {matched}/{len(expected_cols)} matched (+{col_score:.2f})")
        elif not expected_cols:
            score += 0.15
            parts.append("Column check skipped (+0.15)")
        else:
            parts.append("Wrong columns returned")
            
        expected_count = self.task.get("expected_row_count", 0)
        actual_count = len(rows)
        if actual_count == expected_count:
            score += 0.25
            parts.append(f"Row count correct: {actual_count} (+0.25)")
        elif abs(actual_count - expected_count) == 1:
            score += 0.12
            parts.append(f"Row count close: {actual_count} vs {expected_count} (+0.12)")
        else:
            parts.append(f"Row count wrong: {actual_count} vs {expected_count}")
            
        if "order by" in sql_lower:
            score += 0.15
            parts.append("ORDER BY present (+0.15)")
        else:
            parts.append("No ORDER BY")
            
        if "desc" in sql_lower and "order by" in sql_lower:
            score += 0.10
            parts.append("DESC ordering (+0.10)")
            
        return self._clamp(score), " | ".join(parts)

    def _grade_fix(self, sql: str, rows: list, error: Optional[str]) -> Tuple[float, str]:
        score = 0.0
        parts = []
        sql_lower = sql.lower()
        
        if error is None and rows is not None:
            score += 0.35
            parts.append("Query executes successfully (+0.35)")
        else:
            parts.append(f"Still broken: {error}")
            
        expected_count = self.task.get("expected_row_count", 0)
        actual_count = len(rows) if rows else 0
        if actual_count == expected_count:
            score += 0.30
            parts.append(f"Correct rows: {actual_count} (+0.30)")
        elif actual_count > 0:
            score += 0.10
            parts.append(f"Returns data but wrong count: {actual_count} vs {expected_count} (+0.10)")
        else:
            parts.append("Returns no rows")
            
        if "join" in sql_lower and " on " in sql_lower:
            score += 0.20
            parts.append("JOIN...ON syntax present (+0.20)")
        elif "join" in sql_lower:
            parts.append("JOIN present but missing ON keyword")
            
        broken_sql = self.task.get("broken_sql", "")
        if "projects.status" in broken_sql and "projects.status" not in sql:
            score += 0.10
            parts.append("Fixed table alias in WHERE (+0.10)")
        elif "group by" in sql_lower:
            if "group by department" in sql_lower or "group by e.department" in sql_lower:
                score += 0.10
                parts.append("GROUP BY correct column (+0.10)")
                
        return self._clamp(score), " | ".join(parts)

    def _grade_optimize(self, sql: str, rows: list) -> Tuple[float, str]:
        score = 0.0
        parts = []
        sql_upper = sql.upper()
        sql_lower = sql.lower()
        
        uses_cte = "WITH " in sql_upper
        uses_window = any(fn in sql_upper for fn in ["ROW_NUMBER", "RANK(", "DENSE_RANK"])
        if uses_cte or uses_window:
            score += 0.30
            label = "CTE" if uses_cte else "window function"
            parts.append(f"Uses {label} (+0.30)")
        else:
            parts.append("No CTE or window function found")
            
        if "join" in sql_lower:
            score += 0.20
            parts.append("Uses JOIN (+0.20)")
        else:
            parts.append("No JOIN found")
            
        select_count = sql_upper.count("SELECT")
        if select_count == 1:
            score += 0.20
            parts.append("No nested SELECT — no correlated subqueries (+0.20)")
        elif select_count == 2:
            score += 0.10
            parts.append(f"Reduced to {select_count} SELECTs (+0.10)")
        else:
            parts.append(f"Still has {select_count} SELECTs (correlated subqueries remain)")
            
        expected_count = self.task.get("expected_row_count", 0)
        actual_count = len(rows) if rows else 0
        if actual_count == expected_count:
            score += 0.25
            parts.append(f"Correct row count: {actual_count} (+0.25)")
        elif actual_count > 0:
            score += 0.10
            parts.append(f"Returns data, wrong count: {actual_count} vs {expected_count} (+0.10)")
        else:
            parts.append("Returns no rows")
            
        good_patterns = self.task.get("good_patterns", [])
        found_good = [p for p in good_patterns if p.upper() in sql_upper]
        if len(found_good) >= 2:
            score += 0.05
            parts.append(f"Good patterns found: {found_good} (+0.05)")
            
        return self._clamp(score), " | ".join(parts)

    def _build_result(self, reward, feedback, last_sql=None, 
                      last_result=None, last_error=None) -> dict:
        safe_reward = self._clamp(reward)
        observation = {
            "task_id": self.task_id or "",
            "task_type": self.task["type"] if self.task else "",
            "task_description": self.task["description"] if self.task else "",
            "schema_info": get_schema_info(),
            "sample_data": get_sample_data(),
            "expected_description": self._get_hints(),
            "last_sql": last_sql,
            "last_result": last_result,
            "last_error": last_error,
            "step_count": self.step_count,
            "done": self.done,
            "reward": safe_reward,
            "feedback": feedback,
            "metadata": {}
        }
        return {
            "observation": observation,
            "reward": safe_reward,
            "done": self.done,
            "info": {
                "episode_id": self.episode_id,
                "total_reward": round(self.total_reward, 3),
                "best_score": self.best_score,
                "attempts": self.attempts
            }
        }

    def _get_hints(self) -> str:
        if not self.task:
            return ""
        hints = (self.task.get("hints") or self.task.get("optimization_hints") or self.task.get("bugs") or [])
        return "Hints: " + "; ".join(hints) if hints else ""

    @property
    def state(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "task_type": self.task["type"] if self.task else "",
            "task_id": self.task_id or "",
            "step_count": self.step_count,
            "total_reward": round(self.total_reward, 3),
            "best_score_so_far": self.best_score,
            "attempts": self.attempts,
            "done": self.done
        }
