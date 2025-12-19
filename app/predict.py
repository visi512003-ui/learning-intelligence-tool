"""ML prediction module for Learning Intelligence Tool.

Handles model inference, risk detection, and insight generation.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class LearningPredictor:
    """Handles ML predictions for course completion and dropout detection."""
    
    def __init__(self):
        """Initialize predictor with dummy models."""
        self.completion_model = self._create_dummy_model()
        self.dropout_model = self._create_dummy_model()
        self.scaler = StandardScaler()
    
    def _create_dummy_model(self):
        """Create a simple trained model for demonstration."""
        model = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=5)
        X_dummy = np.array([[45, 75, 1], [15, 42, 1], [60, 88, 2], [25, 55, 1]])
        y_dummy = np.array([1, 0, 1, 0])
        model.fit(X_dummy, y_dummy)
        return model
    
    def predict(self, df: pd.DataFrame, features: List[str]) -> List[Dict[str, Any]]:
        """Generate predictions for students."""
        predictions = []
        
        # Handle missing chapter_order
        if 'chapter_order' not in df.columns:
            df['chapter_order'] = 1
        
        for idx, row in df.iterrows():
            try:
                X = np.array([[row['time_spent_min'], row['score_percent'], row.get('chapter_order', 1)]])
                completion_prob = float(self.completion_model.predict_proba(X)[0][1])
                
                # Risk determination
                if completion_prob < 0.3:
                    risk_level = 'HIGH'
                elif completion_prob < 0.6:
                    risk_level = 'MEDIUM'
                else:
                    risk_level = 'LOW'
                
                predictions.append({
                    'student_id': row.get('student_id', f'S{idx}'),
                    'completion_probability': round(completion_prob, 3),
                    'risk_level': risk_level,
                    'predicted_completion': 1 if completion_prob >= 0.5 else 0
                })
            except Exception as e:
                predictions.append({
                    'student_id': row.get('student_id', f'S{idx}'),
                    'error': str(e)
                })
        
        return predictions
    
    def generate_insights(self, df: pd.DataFrame, predictions: List[Dict]) -> Dict[str, Any]:
        """Generate actionable insights from predictions."""
        high_risk = [p for p in predictions if p.get('risk_level') == 'HIGH']
        
        return {
            'high_risk_students': [p['student_id'] for p in high_risk],
            'high_risk_count': len(high_risk),
            'total_students': len(df),
            'key_completion_factors': ['time_spent_min', 'score_percent'],
            'difficult_chapters': [],
            'average_completion_probability': round(np.mean([p['completion_probability'] for p in predictions]), 3),
            'recommendations': 'Focus on high-risk students with personalized intervention'
        }
