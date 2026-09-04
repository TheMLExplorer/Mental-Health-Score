import joblib 
import pandas as pd 
from fastapi import FastAPI 
from pydantic import BaseModel, Field 
from typing import Literal 
 
# Load model 
model = joblib.load("Mental_Health_Model.pkl") 
 
# Countries used by the model 
top_countries = [ 
    "India", "USA", "Canada", "Australia", 
    "UK", "Germany", "Mexico", "Turkey", "France" 
] 
 
app = FastAPI() 
 
 
# Input data 
class StudentData(BaseModel): 
    age: int = Field(..., ge=10, le=100) 
 
    gender: Literal["Male", "Female"] 
 
    country: str 
 
    academic_level: Literal[ 
        "Undergraduate", 
        "Graduate", 
        "High School" 
    ] 
 
    most_used_platform: Literal[ 
        "Facebook", 
        "LinkedIn", 
        "Instagram", 
        "Snapchat", 
        "Twitter", 
        "YouTube", 
        "TikTok", 
        "LINE", 
        "KakaoTalk", 
        "VKontakte", 
        "WhatsApp", 
        "WeChat" 
    ] 
 
    purpose_of_use: Literal[ 
        "Networking", 
        "Education", 
        "Entertainment", 
        "News" 
    ] 
 
    avg_daily_usage_hours: float = Field(..., ge=0, le=24) 
    daily_unlocks: int = Field(..., ge=0) 
    study_hours: float = Field(..., ge=0, le=24) 
    physical_activity_hours: float = Field(..., ge=0, le=24) 
    sleep_hours_per_night: float = Field(..., ge=0, le=24) 
 
    stress_level: Literal[ 
        "Medium", 
        "Low", 
        "Very High", 
        "High" 
    ] 
 
 
# Response 
class PredictionResponse(BaseModel): 
    predicted_mental_health_score: float 
 
 
# Home route 
@app.get("/") 
def greet(): 
    return {"message": "Mental Health Prediction API is running"} 
 
 
# Prediction route 
@app.post("/predict", response_model=PredictionResponse) 
def predict(data: StudentData): 
 
    # Group countries 
    country_group = ( 
        data.country 
        if data.country in top_countries 
        else "Other" 
    ) 
 
    # Create input DataFrame 
    input_row = pd.DataFrame([{ 
        "Age": data.age, 
        "Gender": data.gender, 
        "Country": data.country, 
        "Academic_Level": data.academic_level, 
        "Most_Used_Platform": data.most_used_platform, 
        "Purpose_Of_Use": data.purpose_of_use, 
        "Avg_Daily_Usage_Hours": data.avg_daily_usage_hours, 
        "Daily_Unlocks": data.daily_unlocks, 
        "Study_Hours": data.study_hours, 
        "Physical_Activity_Hours": data.physical_activity_hours, 
        "Sleep_Hours_Per_Night": data.sleep_hours_per_night, 
        "Stress_Level": data.stress_level, 
        "Grouped_country": country_group 
    }]) 
 
    # Prediction 
    prediction = model.predict(input_row)[0] 
 
    # Return prediction 
    return PredictionResponse( 
        predicted_mental_health_score=round(float(prediction), 2) 
    )


# Health check route
@app.get("/health")
def health():
    return {
        "status": "healthy"
    }