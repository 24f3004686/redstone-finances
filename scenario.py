import joblib
import pandas as pd

from financial_engine import (
    cash_health_score,
    stress_probability,
    risk_level
)

# ==========================
# LOAD MODEL
# ==========================

model = joblib.load("runway_model_xgb.pkl")

# ==========================
# CURRENT COMPANY DATA
# ==========================

company = {
    "cash": 1200000,
    "totalOperatingExpenses": 2400000,
    "totalRevenue": 1800000,
    "operatingIncome": 300000,
    "netIncome_x": 200000,
    "totalAssets": 5000000,
    "totalLiab": 2000000,
    "totalCurrentAssets": 2500000,
    "totalCurrentLiabilities": 1000000,
    "changeInCash": 100000,
    "longTermDebt": 500000,
    "netReceivables": 300000
}

def prepare_features(data):

    liquidity_ratio = (
        data["totalCurrentAssets"] /
        data["totalCurrentLiabilities"]
    )

    debt_ratio = (
        data["totalLiab"] /
        data["totalAssets"]
    )

    profit_margin = (
        data["netIncome_x"] /
        data["totalRevenue"]
    )

    return pd.DataFrame([{
        **data,
        "liquidity_ratio": liquidity_ratio,
        "debt_ratio": debt_ratio,
        "profit_margin": profit_margin
    }])


baseline_df = prepare_features(company)

baseline_runway = model.predict(
    baseline_df
)[0]

print("\n===== CURRENT STATE =====")

print(
    "Runway:",
    round(baseline_runway, 2),
    "months"
)

print(
    "Risk:",
    risk_level(baseline_runway)
)


hire_case = company.copy()

hire_case["totalOperatingExpenses"] *= 1.30

hire_df = prepare_features(
    hire_case
)

hire_runway = model.predict(
    hire_df
)[0]

print("\n===== HIRE 5 EMPLOYEES =====")

print(
    "New Runway:",
    round(hire_runway, 2),
    "months"
)

print(
    "Impact:",
    round(
        hire_runway - baseline_runway,
        2
    ),
    "months"
)

print(
    "Risk:",
    risk_level(hire_runway)
)



revenue_case = company.copy()

revenue_case["totalRevenue"] *= 0.70

revenue_df = prepare_features(
    revenue_case
)

revenue_runway = model.predict(
    revenue_df
)[0]

print("\n===== REVENUE DOWN 30% =====")

print(
    "New Runway:",
    round(revenue_runway, 2),
    "months"
)

print(
    "Impact:",
    round(
        revenue_runway - baseline_runway,
        2
    ),
    "months"
)

print(
    "Risk:",
    risk_level(revenue_runway)
)



