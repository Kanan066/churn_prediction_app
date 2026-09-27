from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load model and saved column order
model = joblib.load('model.pkl')
model_columns = joblib.load('model_columns.pkl')

@app.get("/")
def read_root():
    return {"message": "Churn prediction API is running"}

# Define what input data should look like
class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.post("/predict")
def predict_churn(customer: Customer):
    # Convert incoming data to a dataframe
    input_df = pd.DataFrame([customer.dict()])
    
    # Apply the same encoding as training
    input_encoded = pd.get_dummies(input_df)
    
    # Make sure it has the exact same columns as training data (fill missing with 0)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
    
    # Predict
    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]
    
    return {
        "churn_prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": round(float(probability), 3)
    }