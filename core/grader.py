import pandas as pd

class DataFrameGrader:
    """Handles logic for comparing agent SQL result DataFrames against expected DataFrames."""
    
    @staticmethod
    def grade(agent_df: pd.DataFrame, expected_df: pd.DataFrame) -> float:
        """
        Grades a DataFrame result. Returns 0.0 to 1.0.
        Checks for exact match (columns, rows, data types).
        Falls back to unordered match if the structure is the same but sorting is off.
        """
        try:
            agent_norm = agent_df.reset_index(drop=True).sort_index(axis=1)
            expected_norm = expected_df.reset_index(drop=True).sort_index(axis=1)
            
            # Fast exact comparison
            if set(agent_norm.columns) == set(expected_norm.columns):
                
                # Check directly first
                if agent_norm.equals(expected_norm):
                    return 1.0
                
                # Re-align column order if mismatched
                agent_aligned = agent_norm[expected_norm.columns]
                if agent_aligned.equals(expected_norm):
                    return 1.0
                
                # Check unordered row set equivalence
                agent_tuples = set(tuple(row) for row in agent_aligned.to_numpy())
                expected_tuples = set(tuple(row) for row in expected_norm.to_numpy())
                
                if len(agent_tuples) == len(expected_tuples) and agent_tuples == expected_tuples:
                    return 0.8  # Unordered match yields partial points
            
            return 0.0
        except Exception:
            return 0.0
