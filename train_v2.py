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

print("\n===== REDSTONE ANALYZER =====\n")

# ==========================
# USER INPUTS
# ==========================

cash = float(input("Cash Available: "))

total_revenue = float(input("Total Revenue: "))

total_operating_expenses = float(
    input("Total Operating Expenses: ")
)

operating_income = float(
    input("Operating Income: ")
)

net_income = float(
    input("Net Income: ")
)

total_assets = float(
    input("Total Assets: ")
)

total_liab = float(
    input("Total Liabilities: ")
)

total_current_assets = float(
    input("Current Assets: ")
)

total_current_liabilities = float(
    input("Current Liabilities: ")
)

change_in_cash = float(
    input("Change In Cash: ")
)

long_term_debt = float(
    input("Long Term Debt: ")
)

net_receivables = float(
    input("Net Receivables: ")
)

# ==========================
# DERIVED FEATURES
# ==========================

liquidity_ratio = (
    total_current_assets /
    total_current_liabilities
)

debt_ratio = (
    total_liab /
    total_assets
)

profit_margin = (
    net_income /
    total_revenue
)

# ==========================
# CREATE DATAFRAME
# ==========================

sample = pd.DataFrame([{
    "cash": cash,
    "totalOperatingExpenses":
        total_operating_expenses,
    "totalRevenue":
        total_revenue,
    "operatingIncome":
        operating_income,
    "netIncome_x":
        net_income,
    "totalAssets":
        total_assets,
    "totalLiab":
        total_liab,
    "totalCurrentAssets":
        total_current_assets,
    "totalCurrentLiabilities":
        total_current_liabilities,
    "changeInCash":
        change_in_cash,
    "longTermDebt":
        long_term_debt,
    "netReceivables":
        net_receivables,
    "liquidity_ratio":
        liquidity_ratio,
    "debt_ratio":
        debt_ratio,
    "profit_margin":
        profit_margin
}])

# ==========================
# PREDICT
# ==========================

runway = model.predict(sample)[0]

health = cash_health_score(runway)

stress = stress_probability(runway)

risk = risk_level(runway)

# ==========================
# RESULTS
# ==========================

print("\n===== REDSTONE REPORT =====\n")

print(
    "Predicted Runway:",
    round(runway, 2),
    "months"
)

print(
    "Cash Health Score:",
    health,
    "/100"
)

print(
    "Stress Probability:",
    stress,
    "%"
)

print(
    "Risk Level:",
    risk
)
