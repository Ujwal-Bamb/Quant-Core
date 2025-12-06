from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="QuantCore API")

class MarketData(BaseModel):
    price: float
    vix: float
    volume: int

@app.post("/predict")
def predict_move(data: MarketData):
    # In production, this calls the loaded Model Registry
    prediction = 0.55 # Mock prediction
    action = "BUY" if prediction > 0.5 else "HOLD"
    return {
        "probability_up": prediction,
        "suggested_action": action,
        "regime": "LOW_VOL_BULL"
    }

@app.get("/health")
def health():
    return {"status": "operational", "latency_ms": 12}
