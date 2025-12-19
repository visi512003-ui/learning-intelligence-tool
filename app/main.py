"""FastAPI application for Learning Intelligence Tool.

Provides REST API endpoints and CLI interface for course completion prediction,
dropout detection, and learning insights.
"""

import click
import json
import joblib
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import numpy as np
from .preprocess import LearnerDataProcessor
from .predict import LearningPredictor

app = FastAPI(
    title="Learning Intelligence Tool",
    description="AI-powered predictions and insights for educational platforms",
    version="1.0.0"
)

class StudentPrediction(BaseModel):
    student_id: str
    course_id: str
    time_spent_min: float
    score_percent: float
    chapter_order: int = 1

class PredictionResponse(BaseModel):
    status: str
    predictions: List[Dict[str, Any]]
    insights: Dict[str, Any]

# Initialize predictors
predictor = LearningPredictor()

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}

@app.post("/predict")
async def predict_batch(file: UploadFile = File(...)):
    """Batch prediction from CSV file."""
    try:
        contents = await file.read()
        df = pd.read_csv(__import__('io').StringIO(contents.decode()))
        processor = LearnerDataProcessor()
        processed_data, features = processor.process(df)
        predictions = predictor.predict(processed_data, features)
        insights = predictor.generate_insights(df, predictions)
        return {
            "status": "success",
            "predictions": predictions,
            "insights": insights
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict-single")
async def predict_single(student: StudentPrediction):
    """Predict for single student."""
    try:
        data_dict = student.dict()
        df = pd.DataFrame([data_dict])
        processor = LearnerDataProcessor()
        processed_data, features = processor.process(df)
        predictions = predictor.predict(processed_data, features)
        return {
            "status": "success",
            "prediction": predictions[0] if predictions else None
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/insights/{course_id}")
async def get_course_insights(course_id: str):
    """Get aggregated insights for a course."""
    return {
        "course_id": course_id,
        "message": "Load sample data to get insights"
    }

@click.group()
def cli():
    """Learning Intelligence Tool CLI."""
    pass

@cli.command()
@click.option('--input', required=True, help='Input CSV file')
@click.option('--output', required=False, help='Output JSON file')
def predict_cli(input, output):
    """Run predictions via CLI."""
    try:
        df = pd.read_csv(input)
        processor = LearnerDataProcessor()
        processed_data, features = processor.process(df)
        predictions = predictor.predict(processed_data, features)
        insights = predictor.generate_insights(df, predictions)
        
        result = {
            "status": "success",
            "predictions": predictions,
            "insights": insights
        }
        
        if output:
            with open(output, 'w') as f:
                json.dump(result, f, indent=2)
            click.echo(f"Results saved to {output}")
        else:
            click.echo(json.dumps(result, indent=2))
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)

if __name__ == "__main__":
    cli()
