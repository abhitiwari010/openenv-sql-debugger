from dataclasses import dataclass
from typing import List

@dataclass
class SqlTask:
    difficulty: int
    instruction: str
    expected_query: str

class EnvironmentTasks:
    """Registry for all SQL challenges assigned to the OpenEnv agent."""
    def __init__(self):
        self.tasks: List[SqlTask] = [
            SqlTask(
                difficulty=1,
                instruction="EASY: Find all names and salaries of employees, ordering by their salaries descending.",
                expected_query="SELECT NAME, SALARY FROM EMPLOYEES ORDER BY SALARY DESC"
            ),
            SqlTask(
                difficulty=2,
                instruction="MEDIUM: Return the department names alongside the average salary per department. Columns must be 'DEPARTMENT_NAME' and 'AVG_SALARY'.",
                expected_query="""
                    SELECT D.NAME AS DEPARTMENT_NAME, AVG(E.SALARY) AS AVG_SALARY
                    FROM EMPLOYEES E
                    JOIN DEPARTMENTS D ON E.DEPARTMENT_ID = D.ID
                    GROUP BY D.NAME
                """
            ),
            SqlTask(
                difficulty=3,
                instruction="HARD: Find the name and title of all Engineering (DEPARTMENT_ID = 2) employees who earn a 'Senior' level salary (based on ROLES MIN_SALARY limit).",
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
