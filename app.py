from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

from financial_engine import (
    cash_health_score,
    stress_probability,
    risk_level
)

app = FastAPI(
    title="Redstone AI CFO",
    description="AI-powered financial runway and liquidity intelligence platform",
    version="1.0.0"
)

# Load model
model = joblib.load("runway_model_xgb.pkl")


class FinancialInput(BaseModel):
    cash: float
    totalRevenue: float
    totalOperatingExpenses: float
    operatingIncome: float
    netIncome: float
    totalAssets: float
    totalLiabilities: float
    currentAssets: float
    currentLiabilities: float
    changeInCash: float
    longTermDebt: float
    netReceivables: float


class ScenarioInput(BaseModel):

    financials: FinancialInput

    scenario: str

    percentage: float = 0

    amount: float = 0

    employees: int = 0

def prepare_dataframe(data):

    liquidity_ratio = (
        data["currentAssets"] /
        data["currentLiabilities"]
    )

    debt_ratio = (
        data["totalLiabilities"] /
        data["totalAssets"]
    )

    profit_margin = (
        data["netIncome"] /
        data["totalRevenue"]
    )

    return pd.DataFrame([{
        "cash": data["cash"],
        "totalOperatingExpenses":
            data["totalOperatingExpenses"],
        "totalRevenue":
            data["totalRevenue"],
        "operatingIncome":
            data["operatingIncome"],
        "netIncome_x":
            data["netIncome"],
        "totalAssets":
            data["totalAssets"],
        "totalLiab":
            data["totalLiabilities"],
        "totalCurrentAssets":
            data["currentAssets"],
        "totalCurrentLiabilities":
            data["currentLiabilities"],
        "changeInCash":
            data["changeInCash"],
        "longTermDebt":
            data["longTermDebt"],
        "netReceivables":
            data["netReceivables"],
        "liquidity_ratio":
            liquidity_ratio,
        "debt_ratio":
            debt_ratio,
        "profit_margin":
            profit_margin
    }])

def identify_risk_factors(data):

    risks = []

    if data.totalOperatingExpenses > data.totalRevenue:
        risks.append("Operating expenses exceed revenue")

    if data.currentAssets < data.currentLiabilities:
        risks.append("Current liabilities exceed current assets")

    if data.netIncome < 0:
        risks.append("Company is currently loss making")

    if data.longTermDebt > (0.5 * data.totalAssets):
        risks.append("High long-term debt exposure")

    if data.changeInCash < 0:
        risks.append("Negative cash flow trend")

    if len(risks) == 0:
        risks.append("No major financial risks detected")

    return risks

def get_recommendation(runway):

    if runway < 3:
        return (
            "Critical situation. "
            "Raise capital immediately."
        )

    elif runway < 6:
        return (
            "Reduce burn rate and "
            "preserve cash reserves."
        )

    elif runway < 12:
        return (
            "Monitor expenses and "
            "improve profitability."
        )

    return (
        "Financial position appears stable."
    )

@app.get("/")
def home():

    return {
        "product": "Redstone AI CFO",
        "status": "running",
        "model": "XGBoost",
        "version": "1.0.0"
    }


@app.post("/analyze")
def analyze(data: FinancialInput):

    try:

        if data.currentLiabilities == 0:
            raise HTTPException(
                status_code=400,
                detail="Current liabilities cannot be zero"
            )

        if data.totalAssets == 0:
            raise HTTPException(
                status_code=400,
                detail="Total assets cannot be zero"
            )

        if data.totalRevenue == 0:
            raise HTTPException(
                status_code=400,
                detail="Total revenue cannot be zero"
            )

        # Derived Features

        liquidity_ratio = (
            data.currentAssets /
            data.currentLiabilities
        )

        debt_ratio = (
            data.totalLiabilities /
            data.totalAssets
        )

        profit_margin = (
            data.netIncome /
            data.totalRevenue
        )

        # Model Input

        sample = pd.DataFrame([{
            "cash": data.cash,
            "totalOperatingExpenses": data.totalOperatingExpenses,
            "totalRevenue": data.totalRevenue,
            "operatingIncome": data.operatingIncome,
            "netIncome_x": data.netIncome,
            "totalAssets": data.totalAssets,
            "totalLiab": data.totalLiabilities,
            "totalCurrentAssets": data.currentAssets,
            "totalCurrentLiabilities": data.currentLiabilities,
            "changeInCash": data.changeInCash,
            "longTermDebt": data.longTermDebt,
            "netReceivables": data.netReceivables,
            "liquidity_ratio": liquidity_ratio,
            "debt_ratio": debt_ratio,
            "profit_margin": profit_margin
        }])

        # Prediction

        runway = float(
            model.predict(sample)[0]
        )

        health = cash_health_score(runway)

        stress = stress_probability(runway)

        risk = risk_level(runway)

        risk_factors = identify_risk_factors(data)

        return {
            "financial_summary": {
                "runway_months": round(runway, 2),
                "cash_health_score": health,
                "stress_probability": stress,
                "risk_level": risk
            },

            "financial_ratios": {
                "liquidity_ratio": round(liquidity_ratio, 2),
                "debt_ratio": round(debt_ratio, 2),
                "profit_margin": round(profit_margin, 2)
            },

            "top_risk_factors": risk_factors
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.post("/scenario")
def scenario_analysis(data: ScenarioInput):

    company = data.financials.dict()

    scenario_company = company.copy()

    # ==============================
    # CURRENT RUNWAY
    # ==============================

    current_runway = (
        company["cash"] /
        (company["totalOperatingExpenses"] / 12)
    )

    # ==============================
    # REVENUE DROP
    # ==============================

    if data.scenario == "revenue_drop":

        factor = 1 - data.percentage / 100

        scenario_company["totalRevenue"] *= factor
        scenario_company["operatingIncome"] *= factor
        scenario_company["netIncome"] *= factor
        scenario_company["changeInCash"] *= factor

        scenario_company["totalOperatingExpenses"] *= (
            1 + data.percentage / 200
        )

    # ==============================
    # REVENUE GROWTH
    # ==============================

    elif data.scenario == "revenue_growth":

        factor = 1 + data.percentage / 100

        scenario_company["totalRevenue"] *= factor
        scenario_company["operatingIncome"] *= factor
        scenario_company["netIncome"] *= factor
        scenario_company["changeInCash"] *= factor

        scenario_company["totalOperatingExpenses"] *= (
            1 - data.percentage / 400
        )

    # ==============================
    # EXPENSE INCREASE
    # ==============================

    elif data.scenario == "expense_increase":

        factor = 1 + data.percentage / 100

        scenario_company["totalOperatingExpenses"] *= factor

    # ==============================
    # HIRING
    # ==============================

    elif data.scenario == "hiring":

        employee_cost = 150000

        extra_cost = (
            data.employees *
            employee_cost
        )

        scenario_company["totalOperatingExpenses"] += extra_cost

    # ==============================
    # INVESTMENT
    # ==============================

    elif data.scenario == "investment":

        scenario_company["cash"] += data.amount

    # ==============================
    # DOWNTURN
    # ==============================

    elif data.scenario == "downturn":

        scenario_company["cash"] *= 0.85
        scenario_company["totalRevenue"] *= 0.70
        scenario_company["operatingIncome"] *= 0.50
        scenario_company["netIncome"] *= 0.50

        scenario_company["totalOperatingExpenses"] *= 1.20

    else:

        raise HTTPException(
            status_code=400,
            detail="Invalid scenario"
        )

    # ==============================
    # PROJECTED RUNWAY
    # ==============================

    projected_runway = (
        scenario_company["cash"] /
        (
            scenario_company[
                "totalOperatingExpenses"
            ] / 12
        )
    )

    return {

        "scenario": data.scenario,

        "current_runway":
            round(current_runway, 2),

        "projected_runway":
            round(projected_runway, 2),

        "impact_months":
            round(
                projected_runway -
                current_runway,
                2
            ),

        "current_risk":
            risk_level(current_runway),

        "projected_risk":
            risk_level(projected_runway),

        "recommendation":
            get_recommendation(
                projected_runway
            )
    }
