from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from typing import Optional
import csv
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
import logging
from utils import make_hard_prediction, make_soft_prediction

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReviewRequest(BaseModel):
    review: str

class HeatmapRequest(BaseModel):
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    crime_type: Optional[int] = None

@app.post('/sentiment_score')
async def get_sentiment_score(request: ReviewRequest):
    inputs = request.review.split(',')
    try:
        inputs = list(map(float, inputs))
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid input format. All inputs must be floats.")
    
    score = make_soft_prediction(inputs)

    response = {
        'review': inputs,
        'score': float(score)
    }

    logger.info("response is: %s", response)

    return response

crime_data = pd.read_csv('nagpur_crime_data.csv')

@app.post("/heatmap-data")
async def get_heatmap_data(request: HeatmapRequest):
    try:
        filtered_data = crime_data.copy()
        
        # Apply filters
        if request.start_time:
            filtered_data = filtered_data[
                filtered_data['timestamp'] >= request.start_time
            ]
        
        if request.end_time:
            filtered_data = filtered_data[
                filtered_data['timestamp'] <= request.end_time
            ]
            
        if request.crime_type is not None:
            filtered_data = filtered_data[
                filtered_data['crime_type'] == request.crime_type
            ]
        
        # Prepare heatmap data
        heatmap_data = filtered_data[['latitude', 'longitude']].values.tolist()
        
        return {
            "heatmap_points": heatmap_data,
            "total_crimes": len(heatmap_data)
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)