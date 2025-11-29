from fastapi.testclient import TestClient
from app import app
import pytest

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hotel Booking Prediction API is running."}

# Mock model loading would typically go here for robust testing
# For simplicity, we assume the model file exists or we skip if not
def test_predict_endpoint():
    payload = {
        "no_of_adults": 2,
        "no_of_children": 0,
        "no_of_weekend_nights": 1,
        "no_of_week_nights": 2,
        "type_of_meal_plan": "Meal Plan 1",
        "required_car_parking_space": 0,
        "room_type_reserved": "Room_Type 1",
        "lead_time": 50,
        "arrival_year": 2018,
        "arrival_month": 10,
        "arrival_date": 2,
        "market_segment_type": "Online",
        "repeated_guest": 0,
        "no_of_previous_cancellations": 0,
        "no_of_previous_bookings_not_canceled": 0,
        "avg_price_per_room": 100.0,
        "no_of_special_requests": 0
    }
    
    # We allow 503 if model isn't trained yet in CI environment
    response = client.post("/predict", json=payload)
    if response.status_code == 503:
        pytest.skip("Model not trained yet")
    
    assert response.status_code == 200
    assert "prediction" in response.json()