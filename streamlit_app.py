from pathlib import Path
import joblib
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="User Churn Predictor with Intervention Recommender",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "churn_logreg_pipeline.joblib"
API_URL = "http://127.0.0.1:8000/predict"


@st.cache_resource
def load_local_model():
  return joblib.load(MODEL_PATH)


def recommend_row(row_dict: dict, prob: float) -> tuple[str, str]:
  actions = []
  if row_dict.get("Contract") == "Month-to-month":
    actions.append("Offer annual/two-year contract discount or lock-in perk")
  if (
      row_dict.get("InternetService") == "Fiber optic"
      and row_dict.get("TechSupport") != "Yes"
  ):
    actions.append("Provide onboarding check-in + TechSupport trial")
  if float(row_dict.get("tenure", 99)) <= 12:
    actions.append("Trigger early-tenure check-in / loyalty credit")
  if float(row_dict.get("MonthlyCharges", 0)) >= 75 and not actions:
    actions.append("Review plan bundle / bill optimization")
  if not actions:
    actions.append("Maintain standard engagement")

  if prob >= 0.60:
    tier = "High"
  elif prob >= 0.35:
    tier = "Medium"
  else:
    tier = "Low"
  return tier, " | ".join(actions)


st.title("User Churn Predictor with Intervention Recommender")
st.caption(
    "Logistic Regression Pipeline (ROC-AUC: 0.8359) + FastAPI + Streamlit"
)

tab1, tab2 = st.tabs(["Single Customer Prediction", "Batch CSV Scoring"])

with tab1:
  col1, col2, col3 = st.columns(3)

  with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (Months)", min_value=1, max_value=72, value=4)
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])

  with col2:
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox(
        "Multiple Lines", ["No", "Yes", "No phone service"]
    )
    InternetService = st.selectbox(
        "Internet Service", ["Fiber optic", "DSL", "No"]
    )
    OnlineSecurity = st.selectbox(
        "Online Security", ["No", "Yes", "No internet service"]
    )
    OnlineBackup = st.selectbox(
        "Online Backup", ["No", "Yes", "No internet service"]
    )
    DeviceProtection = st.selectbox(
        "Device Protection", ["No", "Yes", "No internet service"]
    )

  with col3:
    TechSupport = st.selectbox(
        "Tech Support", ["No", "Yes", "No internet service"]
    )
    StreamingTV = st.selectbox(
        "Streaming TV", ["No", "Yes", "No internet service"]
    )
    StreamingMovies = st.selectbox(
        "Streaming Movies", ["No", "Yes", "No internet service"]
    )
    Contract = st.selectbox(
        "Contract", ["Month-to-month", "One year", "Two year"]
    )
    PaymentMethod = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )
    MonthlyCharges = st.number_input(
        "Monthly Charges ($)", min_value=0.0, max_value=200.0, value=89.5
    )
    TotalCharges = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        max_value=15000.0,
        value=float(tenure * 89.5),
    )

  if st.button("Predict Churn Risk", type="primary"):
    payload = {
        "gender": gender,
        "SeniorCitizen": int(SeniorCitizen),
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": int(tenure),
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": float(MonthlyCharges),
        "TotalCharges": float(TotalCharges),
    }

    try:
      resp = requests.post(API_URL, json=payload, timeout=5)
      resp.raise_for_status()
      result = resp.json()
      prob = result["churn_probability"]
      tier = result["risk_tier"]
      actions_text = " | ".join(result["recommended_actions"])
      source_note = "Scored via FastAPI (`/predict`)"
    except Exception:
      pipe = load_local_model()
      prob = float(pipe.predict_proba(pd.DataFrame([payload]))[0, 1])
      tier, actions_text = recommend_row(payload, prob)
      source_note = "Scored via local pipeline fallback"

    m1, m2, m3 = st.columns(3)
    m1.metric("Churn Probability", f"{prob:.2%}")
    m2.metric("Risk Tier", tier)
    m3.metric("Predicted Churn", "Yes" if prob >= 0.50 else "No")

    st.info(f"**Recommended Actions:** {actions_text}")
    st.caption(source_note)

with tab2:
  uploaded = st.file_uploader(
      "Upload Customer CSV (with Telco feature columns)", type=["csv"]
  )
  if uploaded is not None:
    batch_df = pd.read_csv(uploaded)
    pipe = load_local_model()
    feature_cols = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
    ]
    clean_batch = batch_df.copy()
    clean_batch["TotalCharges"] = pd.to_numeric(
        clean_batch["TotalCharges"], errors="coerce"
    ).fillna(0.0)
    probs = pipe.predict_proba(clean_batch[feature_cols])[:, 1]
    tiers_and_actions = [
        recommend_row(rec, float(p))
        for rec, p in zip(clean_batch[feature_cols].to_dict("records"), probs)
    ]
    out_df = batch_df.copy()
    out_df["churn_probability"] = probs.round(4)
    out_df["risk_tier"] = [t for t, _ in tiers_and_actions]
    out_df["recommended_actions"] = [a for _, a in tiers_and_actions]
    st.dataframe(
        out_df.sort_values("churn_probability", ascending=False),
        use_container_width=True,
    )
    st.download_button(
        "Download Scored CSV",
        out_df.to_csv(index=False).encode("utf-8"),
        file_name="scored_telco_customers.csv",
        mime="text/csv",
    )