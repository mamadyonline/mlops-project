from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
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
        
        # Download deployed model from S3
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            s3_client.download_file(
                'mlflow-project-artifacts-remote',
                'deployment_artifacts/model.pkl',
                tmp_file.name
            )
            
            model = joblib.load(tmp_file.name)
            os.unlink(tmp_file.name)
            
        return model
    except Exception as e:
        print(f"Error loading model from S3: {e}")
        raise

app = FastAPI(
    title="Heart Disease Prediction API",
    description="API for predicting heart disease risk",
    version="1.0.0",
    lifespan=lifespan
)

# Pydantic models for request/response
class PredictionInput(BaseModel):
    """Patient data model for heart disease risk prediction"""
    age: int = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex (0=female, 1=male)")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: int = Field(..., ge=50, le=250, description="Resting blood pressure")
    chol: int = Field(..., ge=100, le=600, description="Cholesterol level")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: int = Field(..., ge=50, le=250, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment")
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels colored by fluoroscopy")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia type")

class BatchPredictionInput(BaseModel):
    data: List[PredictionInput]

class PredictionOutput(BaseModel):
    prediction: int = Field(..., description="Binary prediction (0=no risk, 1=risk)")
    probability: float = Field(..., description="Probability of heart disease (0-1)")
    risk_level: str = Field(..., description="Risk level: Low, Medium, High")

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
        input_df = pd.DataFrame([input_data.model_dump()])
        
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
        input_df = pd.DataFrame([item.dict() for item in input_data.data])
        
        predictions = model.predict(input_df)
        probabilities = model.predict_proba(input_df)[:, 1]  # Probability of positive class
        
        results = []
        for pred, prob in zip(predictions, probabilities):
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