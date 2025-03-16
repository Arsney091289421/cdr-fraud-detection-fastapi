from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel, Field

# 1️⃣ 加载模型（没有特征名，只加载模型本身）
rf_model = joblib.load("random_forest_model.pkl")

# 2️⃣ **同步更新训练时的列名，使用带下划线的格式**
expected_columns = [
    "Account_Length", "VMail_Message", "Day_Mins", "Day_Calls", "Day_Charge",
    "Eve_Mins", "Eve_Calls", "Eve_Charge", "Night_Mins", "Night_Calls", "Night_Charge",
    "Intl_Mins", "Intl_Calls", "Intl_Charge", "CustServ_Calls"
]

# 3️⃣ 创建 FastAPI 实例
app = FastAPI()

# 4️⃣ 定义输入数据格式（API 仍然支持小写 JSON，但内部转换为训练列名）
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

# 5️⃣ 预测 API 端点
@app.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([data.model_dump(by_alias=True)])  # 转换 DataFrame
    
    # **确保 API 传入的列顺序和训练时一致**
    df = df[expected_columns]  # 重新排序 DataFrame
    
    prediction = rf_model.predict(df)[0]  # 进行预测
    return {"isFraud": int(prediction)}

# 6️⃣ 运行 API 服务器
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
