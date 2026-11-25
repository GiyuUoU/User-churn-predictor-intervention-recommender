from pathlib import Path
from typing import Literal
from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "churn_logreg_pipeline.joblib"

app = FastAPI(
    title="User Churn Predictor with Intervention Recommender API",
    description="Predicts user churn probability, risk tier, and retention actions.",
    version="1.0.0",
)

model_pipe = joblib.load(MODEL_PATH)


class CustomerInput(BaseModel):
  gender: Literal["Male", "Female"]
  SeniorCitizen: int = Field(ge=0, le=1)
  Partner: Literal["Yes", "No"]
  Dependents: Literal["Yes", "No"]
  tenure: int = Field(ge=0, le=120)
  PhoneService: Literal["Yes", "No"]
  MultipleLines: Literal["Yes", "No", "No phone service"]
  InternetService: Literal["DSL", "Fiber optic", "No"]
  OnlineSecurity: Literal["Yes", "No", "No internet service"]
  OnlineBackup: Literal["Yes", "No", "No internet service"]
  DeviceProtection: Literal["Yes", "No", "No internet service"]
  TechSupport: Literal["Yes", "No", "No internet service"]
  StreamingTV: Literal["Yes", "No", "No internet service"]
  StreamingMovies: Literal["Yes", "No", "No internet service"]
  Contract: Literal["Month-to-month", "One year", "Two year"]
  PaperlessBilling: Literal["Yes", "No"]
  PaymentMethod: Literal[
      "Electronic check",
      "Mailed check",
      "Bank transfer (automatic)",
      "Credit card (automatic)",
  ]
  MonthlyCharges: float = Field(ge=0)
  TotalCharges: float = Field(ge=0)


def build_recommendations(row: dict, prob: float) -> tuple[str, list[str]]:
  actions = []
  if row.get("Contract") == "Month-to-month":
    actions.append("Offer annual/two-year contract discount or lock-in perk")
  if (
      row.get("InternetService") == "Fiber optic"
      and row.get("TechSupport") != "Yes"
  ):
    actions.append("Provide onboarding check-in + TechSupport trial")
  if row.get("tenure", 99) <= 12:
    actions.append("Trigger early-tenure check-in / loyalty credit")
  if row.get("MonthlyCharges", 0) >= 75 and not actions:
    actions.append("Review plan bundle / bill optimization")
  if not actions:
    actions.append("Maintain standard engagement")

  if prob >= 0.60:
    risk_tier = "High"
  elif prob >= 0.35:
    risk_tier = "Medium"
  else:
    risk_tier = "Low"

  return risk_tier, actions


@app.get("/health")
def health_check():
  return {"status": "ok", "model_loaded": model_pipe is not None}


@app.post("/predict")
def predict_churn(customer: CustomerInput):
  try:
    payload = customer.model_dump()
    df_input = pd.DataFrame([payload])
    prob = float(model_pipe.predict_proba(df_input)[0, 1])
    pred_label = "Yes" if prob >= 0.50 else "No"
    risk_tier, actions = build_recommendations(payload, prob)
    return {
        "churn_prediction": pred_label,
        "churn_probability": round(prob, 4),
        "risk_tier": risk_tier,
        "recommended_actions": actions,
    }
  except Exception as exc:
    raise HTTPException(status_code=500, detail=str(exc)) from exc