import joblib
import lightgbm as lgb
import pandas as pd
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel

# Define the input data schema
class PredictionInput(BaseModel):
    step: float
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    isFlaggedFraud: int

# Define the output data schema
class PredictionOutput(BaseModel):
    predictions: list = []

# Load the trained model
model = joblib.load('lgb_model.pkl')

# Create the FastAPI application
app = FastAPI()

# Define the prediction endpoint
@app.post("/predict")
def predict(data: PredictionInput):
    # Convert the input data to a pandas DataFrame
    input_data = pd.DataFrame([data.dict()])

    # Make predictions using the loaded model
    predictions = model.predict(input_data)

    # Return the predictions as a JSON response
    return {"predictions": predictions.tolist()}

# Generate the OpenAPI schema with Swagger UI support
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Fraud Detection Model",
        version="1.0.0",
        description="Light GBM model",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Run the FastAPI application with Swagger UI enabled
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)