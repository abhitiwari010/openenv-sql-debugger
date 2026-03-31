from dataclasses import dataclass
from typing import List

@dataclass
class SqlTask:
    task_id: str
    difficulty: int
    difficulty_label: str
    instruction: str
    expected_query: str

class EnvironmentTasks:
    """Registry for all SQL challenges assigned to the OpenEnv agent."""
    def __init__(self):
        self.tasks: List[SqlTask] = [
            SqlTask(
                task_id="employee_payroll_overview",
                difficulty=1,
                difficulty_label="easy",
                instruction="EASY: As an HR analyst, list all employee names and salaries ordered by salary descending for payroll review.",
                expected_query="SELECT NAME, SALARY FROM EMPLOYEES ORDER BY SALARY DESC"
            ),
            SqlTask(
                task_id="department_budget_summary",
                difficulty=2,
                difficulty_label="medium",
                instruction="MEDIUM: As a finance analyst, return each department name with average salary. Output columns must be 'DEPARTMENT_NAME' and 'AVG_SALARY'.",
                expected_query="""
                    SELECT D.NAME AS DEPARTMENT_NAME, AVG(E.SALARY) AS AVG_SALARY
                    FROM EMPLOYEES E
                    JOIN DEPARTMENTS D ON E.DEPARTMENT_ID = D.ID
                    GROUP BY D.NAME
                """
            ),
            SqlTask(
                task_id="senior_engineering_comp_review",
                difficulty=3,
                difficulty_label="hard",
                instruction="HARD: For compensation review, find Engineering employees whose salary qualifies for the 'Senior' role threshold. Return employee NAME and role TITLE.",
                expected_query="""
                    SELECT E.NAME, R.TITLE 
                    FROM EMPLOYEES E
                    JOIN ROLES R ON E.SALARY >= R.MIN_SALARY
                    WHERE E.DEPARTMENT_ID = 2 AND R.TITLE = 'Senior'
                """
            )
        ]
        
    def get_total_tasks(self) -> int:
        return len(self.tasks)
        
    def get_task(self, index: int) -> SqlTask:
        return self.tasks[index]
