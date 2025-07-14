# app.py - FastAPI application for model serving
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import boto3
import os
from typing import List, Dict, Any
import tempfile
from contextlib import asynccontextmanager

# Global variable to store the model
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup"""
    global model
    try:
        # Load model from S3 on startup
        model = load_model_from_s3()
        print("✓ Model loaded successfully")
        yield
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    finally:
        # Clean up if needed
        pass

def load_model_from_s3():
    """Load the trained model from S3"""
    try:
        # Initialize S3 client
        s3_client = boto3.client('s3')
        
        # Download model from S3
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            s3_client.download_file(
                'mlflow-project-artifacts-remote',
                'deployment_artifacts/model.pkl',
                tmp_file.name
            )
            
            # Load model
            model = joblib.load(tmp_file.name)
            
            # Clean up
            os.unlink(tmp_file.name)
            
        return model
    except Exception as e:
        print(f"Error loading model from S3: {e}")
        raise

# Create FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    description="API for predicting heart disease risk",
    version="1.0.0",
    lifespan=lifespan
)

# Pydantic models for request/response
class PredictionInput(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float
    
    class Config:
        schema_extra = {
            "example": {
                "age": 63,
                "sex": 1,
                "cp": 3,
                "trestbps": 145,
                "chol": 233,
                "fbs": 1,
                "restecg": 0,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 0,
                "ca": 0,
                "thal": 1
            }
        }

class BatchPredictionInput(BaseModel):
    data: List[PredictionInput]

class PredictionOutput(BaseModel):
    prediction: int
    probability: float
    risk_level: str

class BatchPredictionOutput(BaseModel):
    predictions: List[PredictionOutput]

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Heart Disease Prediction API is running"}

@app.get("/health")
async def health():
    """Health check endpoint"""
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """Make a single prediction"""
    global model
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data.model_dump()])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]  # Probability of positive class
        
        # Determine risk level
        # this is arbitrary and may need tuning 
        # to literature values 
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return PredictionOutput(
            prediction=int(prediction),
            probability=float(probability),
            risk_level=risk_level
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionOutput)
async def predict_batch(input_data: BatchPredictionInput):
    """Make batch predictions"""
    global model
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([item.dict() for item in input_data.data])
        
        # Make predictions
        predictions = model.predict(input_df)
        probabilities = model.predict_proba(input_df)[:, 1]  # Probability of positive class
        
        # Create response
        results = []
        for pred, prob in zip(predictions, probabilities):
            # Determine risk level
            if prob < 0.3:
                risk_level = "Low"
            elif prob < 0.7:
                risk_level = "Medium"
            else:
                risk_level = "High"
            
            results.append(PredictionOutput(
                prediction=int(pred),
                probability=float(prob),
                risk_level=risk_level
            ))
        
        return BatchPredictionOutput(predictions=results)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Get model information"""
    global model
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": type(model).__name__,
        "features": model.feature_names_in_.tolist() if hasattr(model, 'feature_names_in_') else "Unknown",
        "n_features": model.n_features_in_ if hasattr(model, 'n_features_in_') else "Unknown"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)