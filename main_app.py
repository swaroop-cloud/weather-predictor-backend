from datetime import date

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from prediction import predict_tomorrow

app = FastAPI(title="Next-Day Weather Risk Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your Netlify URL once deployed
    allow_methods=["*"],
    allow_headers=["*"],
)


class DayInput(BaseModel):
    date: date
    tmax: float
    tmin: float
    prcp: float = Field(ge=0)


class PredictRequest(BaseModel):
    days: list[DayInput]  # exactly 8, oldest first, today last
    snwd_yesterday: float = 0.0


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Next-day weather predictor API is running."}


@app.post("/predict")
def predict(payload: PredictRequest):
    if len(payload.days) != 8:
        raise HTTPException(status_code=400, detail="Provide exactly 8 consecutive days (oldest first, today last).")

    days = [
        {"date": d.date.isoformat(), "tmax": d.tmax, "tmin": d.tmin, "prcp": d.prcp}
        for d in payload.days
    ]

    try:
        return predict_tomorrow(days, snwd_yesterday=payload.snwd_yesterday)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
