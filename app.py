from fastapi import FastAPI, Query, Request
import joblib
import pandas as pd
from pydantic import BaseModel, Field
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse


rf_model = joblib.load("random_forest_model.pkl")

expected_columns = [
    "Account_Length", "VMail_Message", "Day_Mins", "Day_Calls", "Day_Charge",
    "Eve_Mins", "Eve_Calls", "Eve_Charge", "Night_Mins", "Night_Calls", "Night_Charge",
    "Intl_Mins", "Intl_Calls", "Intl_Charge", "CustServ_Calls"
]

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


class InputData(BaseModel):
    account_length: int = Field(..., alias="Account_Length")
    vmail_message: int = Field(..., alias="VMail_Message")
    day_mins: float = Field(..., alias="Day_Mins")
    day_calls: int = Field(..., alias="Day_Calls")
    day_charge: float = Field(..., alias="Day_Charge")
    eve_mins: float = Field(..., alias="Eve_Mins")
    eve_calls: int = Field(..., alias="Eve_Calls")
    eve_charge: float = Field(..., alias="Eve_Charge")
    night_mins: float = Field(..., alias="Night_Mins")
    night_calls: int = Field(..., alias="Night_Calls")
    night_charge: float = Field(..., alias="Night_Charge")
    intl_mins: float = Field(..., alias="Intl_Mins")
    intl_calls: int = Field(..., alias="Intl_Calls")
    intl_charge: float = Field(..., alias="Intl_Charge")
    custserv_calls: int = Field(..., alias="CustServ_Calls")

    class Config:
        populate_by_name = True

@app.post("/predict")
def predict(data: InputData, threshold: float = Query(default=0.46, description="Fraud threshold")):
    df = pd.DataFrame([data.model_dump(by_alias=True)])
    df = df[expected_columns]
    
    fraud_proba = rf_model.predict_proba(df)[0][1]
    is_fraud = int(fraud_proba >= threshold)
    
    return {
        "fraud_detected": bool(is_fraud),
        "fraud_probability": f"{fraud_proba * 100:.2f}%",
        "threshold_used": threshold
    }

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/ui")
def serve_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/")
def root():
    return RedirectResponse(url="/ui")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
