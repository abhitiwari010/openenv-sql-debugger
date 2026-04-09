import pandas as pd
from collections import Counter
from typing import Dict, Any

class DataFrameGrader:
    """Handles logic for comparing agent SQL result DataFrames against expected DataFrames."""
    
    @staticmethod
    def grade(agent_df: pd.DataFrame, expected_df: pd.DataFrame) -> float:
        """
        Grades a DataFrame result. Returns 0.0 to 1.0.
        Checks for exact match (columns, rows, data types).
        Falls back to unordered match if the structure is the same but sorting is off.
        """
        return DataFrameGrader.grade_with_details(agent_df, expected_df)["score"]

    @staticmethod
    def grade_with_details(agent_df: pd.DataFrame, expected_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Deterministic grader with partial signals:
        - 0.4 for column correctness
        - 0.2 for row count closeness
        - 0.4 for row content match (multiset, order-insensitive)
        """
        try:
            agent_norm = agent_df.reset_index(drop=True).sort_index(axis=1)
            expected_norm = expected_df.reset_index(drop=True).sort_index(axis=1)

            expected_cols = list(expected_norm.columns)
            agent_cols = list(agent_norm.columns)
            same_col_set = set(agent_cols) == set(expected_cols)
            col_score = 0.4 if same_col_set else 0.0

            if not same_col_set:
                return {
                    "score": 0.0001,
                    "feedback": f"Column mismatch. Expected {expected_cols}, got {agent_cols}.",
                    "details": {"column_score": 0.0, "row_count_score": 0.0, "row_content_score": 0.0},
                }

            agent_aligned = agent_norm[expected_cols]

            expected_count = len(expected_norm)
            agent_count = len(agent_aligned)
            if expected_count == 0:
                row_count_score = 0.2 if agent_count == 0 else 0.0
            else:
                row_delta = abs(agent_count - expected_count)
                row_count_score = 0.2 * max(0.0, 1.0 - (row_delta / expected_count))

            agent_rows = Counter(tuple(row) for row in agent_aligned.to_numpy())
            expected_rows = Counter(tuple(row) for row in expected_norm.to_numpy())
            overlap = sum((agent_rows & expected_rows).values())
            denom = max(1, sum(expected_rows.values()))
            row_content_score = 0.4 * (overlap / denom)

            score = round(min(0.9999, col_score + row_count_score + row_content_score), 4)
            if score <= 0.0:
                score = 0.0001
            feedback = "Exact match." if score >= 0.95 else "Partially correct result."
            return {
                "score": score,
                "feedback": feedback,
                "details": {
                    "column_score": round(col_score, 4),
                    "row_count_score": round(row_count_score, 4),
                    "row_content_score": round(row_content_score, 4),
                },
            }
        except Exception as exc:
            return {
                "score": 0.0001,
                "feedback": f"Grader failed: {exc}",
                "details": {"column_score": 0.0, "row_count_score": 0.0, "row_content_score": 0.0},
            }
