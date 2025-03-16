from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel, Field


rf_model = joblib.load("random_forest_model.pkl")


expected_columns = [
    "Account_Length", "VMail_Message", "Day_Mins", "Day_Calls", "Day_Charge",
    "Eve_Mins", "Eve_Calls", "Eve_Charge", "Night_Mins", "Night_Calls", "Night_Charge",
    "Intl_Mins", "Intl_Calls", "Intl_Charge", "CustServ_Calls"
]


app = FastAPI()


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
def predict(data: InputData):
    df = pd.DataFrame([data.model_dump(by_alias=True)])  
    
    
    df = df[expected_columns]  
    
    prediction = rf_model.predict(df)[0]  
    return {"isFraud": int(prediction)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
