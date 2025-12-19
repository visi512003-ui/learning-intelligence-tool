"""Data preprocessing module for Learning Intelligence Tool.

Handles feature engineering, data validation, and preparation.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List

class LearnerDataProcessor:
    """Preprocesses and engineers features from learner data."""
    
    REQUIRED_COLUMNS = ['student_id', 'course_id', 'time_spent_min', 'score_percent']
    
    def __init__(self):
        self.column_mapping = {
            'time_spent_min': 'time_spent',
            'score_percent': 'score'
        }
    
    def validate_input(self, df: pd.DataFrame) -> bool:
        """Validate input data has required columns."""
        missing = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        if df.empty:
            raise ValueError("Empty dataframe provided")
        return True
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features for modeling."""
        df = df.copy()
        
        # Engagement score: weighted combination of time and performance
        df['engagement_score'] = (df['time_spent_min'] / 60) * (df['score_percent'] / 100)
        
        # Time to score ratio
        df['time_score_ratio'] = df['time_spent_min'] / (df['score_percent'] + 1)
        
        # Performance level
        df['perf_level'] = pd.cut(df['score_percent'], bins=[0, 40, 70, 100], labels=['low', 'medium', 'high'])
        
        # Time engagement level
        df['time_level'] = pd.cut(df['time_spent_min'], bins=[0, 30, 60, np.inf], labels=['low', 'medium', 'high'])
        
        return df
    
    def process(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Process input data and return processed features."""
        self.validate_input(df)
        df_processed = self.engineer_features(df)
        
        feature_columns = ['time_spent_min', 'score_percent', 'engagement_score', 'time_score_ratio']
        if 'chapter_order' in df_processed.columns:
            feature_columns.append('chapter_order')
        
        return df_processed, feature_columns
