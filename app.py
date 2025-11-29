import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from feature_engineering import create_features

# Load the trained model at startup
model_path = "best_hotel_booking_model.joblib"
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    model = None
    print("Warning: Model file not found. API will not be able to predict until model is trained.")

app = FastAPI(title="Hotel Booking Cancellation Predictor")

# Define input schema matching your CSV columns
class BookingRequest(BaseModel):
    no_of_adults: int
    no_of_children: int
    no_of_weekend_nights: int
    no_of_week_nights: int
    type_of_meal_plan: str
    required_car_parking_space: int
    room_type_reserved: str
    lead_time: int
    arrival_year: int
    arrival_month: int
    arrival_date: int
    market_segment_type: str
    repeated_guest: int
    no_of_previous_cancellations: int
    no_of_previous_bookings_not_canceled: int
    avg_price_per_room: float
    no_of_special_requests: int

@app.get("/")
def home():
    return {"message": "Hotel Booking Prediction API is running."}

@app.post("/predict")
def predict_cancellation(booking: BookingRequest):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded. Train model first.")
    
    try:
        # 1. Convert input JSON to DataFrame
        data = booking.dict()
        df = pd.DataFrame([data])
        
        # 2. Apply Feature Engineering (Crucial: Must match training logic)
        # Note: We reuse the function from your ML pipeline to ensure consistency
        df = create_features(df)
        
        # 3. Predict
        # The loaded model pipeline handles encoding/scaling internally
        prediction = model.predict(df)
        probability = model.predict_proba(df)[:, 1] if hasattr(model, "predict_proba") else None
        
        result = "Canceled" if prediction[0] == 1 else "Not_Canceled"
        
        return {
            "prediction": result,
            "cancellation_probability": float(probability[0]) if probability is not None else "N/A"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)