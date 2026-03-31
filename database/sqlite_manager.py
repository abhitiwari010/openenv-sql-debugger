import sqlite3
import pandas as pd
from typing import List, Tuple

class SqliteManager:
    """Manages the lifecycle and execution of an internal SQLite database."""
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.conn = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def connect(self):
        """Establishes the connection and sets up initial dummy schema + data."""
        self.conn = sqlite3.Connection(self.db_path)
        self._initialize_schema()

    def close(self):
        if self.conn:
            self.conn.close()

    def _initialize_schema(self):
        cursor = self.conn.cursor()
        
        cursor.execute('''CREATE TABLE DEPARTMENTS (ID INTEGER PRIMARY KEY, NAME TEXT)''')
        cursor.execute('''CREATE TABLE EMPLOYEES (ID INTEGER PRIMARY KEY, NAME TEXT, DEPARTMENT_ID INTEGER, SALARY REAL)''')
        cursor.execute('''CREATE TABLE ROLES (ID INTEGER PRIMARY KEY, TITLE TEXT, MIN_SALARY REAL)''')
        
        cursor.executemany("INSERT INTO DEPARTMENTS (ID, NAME) VALUES (?, ?)", [
            (1, "HR"), (2, "Engineering"), (3, "Marketing"), (4, "Sales")
        ])
        
        cursor.executemany("INSERT INTO EMPLOYEES (ID, NAME, DEPARTMENT_ID, SALARY) VALUES (?, ?, ?, ?)", [
            (1, "Alice", 2, 120000), (2, "Bob", 2, 110000), (3, "Charlie", 1, 80000),
            (4, "David", 3, 90000), (5, "Eve", 4, 95000), (6, "Frank", 2, 105000),
            (7, "Grace", 4, 100000), (8, "Hank", 2, 85000)
        ])
        
        cursor.executemany("INSERT INTO ROLES (ID, TITLE, MIN_SALARY) VALUES (?, ?, ?)", [
            (1, "Associate", 60000), (2, "Senior", 100000), (3, "Staff", 130000)
        ])
        
        self.conn.commit()

    def get_schema_summary(self) -> str:
        return """
Database Schema:
1. DEPARTMENTS (ID INTEGER PRIMARY KEY, NAME TEXT)
2. EMPLOYEES (ID INTEGER PRIMARY KEY, NAME TEXT, DEPARTMENT_ID INTEGER REFERENCES DEPARTMENTS(ID), SALARY REAL)
3. ROLES (ID INTEGER PRIMARY KEY, TITLE TEXT, MIN_SALARY REAL)
"""

    def is_safe_query(self, query: str) -> bool:
        """Validates that a query isn't destructive."""
        forbidden = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER"]
        upper_query = query.upper()
        return not any(f in upper_query for f in forbidden)

    def execute_dataframe(self, query: str) -> pd.DataFrame:
        """Executes a SQL query and returns its result as a Pandas DataFrame."""
        if not self.conn:
            raise RuntimeError("Database connection not initialized.")
        if not self.is_safe_query(query):
            raise ValueError("Destructive operations are not allowed.")
        return pd.read_sql_query(query, self.conn)
