import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

from xgboost import XGBRegressor

# ==========================
# LOAD DATA
# ==========================

df = pd.read_csv("master_dataset.csv")

print("Original Shape:", df.shape)

# ==========================
# CREATE TARGET
# ==========================

df["monthly_burn"] = (
    df["totalOperatingExpenses"] / 12
)

df = df[df["monthly_burn"] > 0]

df["runway_target"] = (
    df["cash"] /
    df["monthly_burn"]
)

# Remove extreme outliers

df = df[df["runway_target"] < 60]

print("\nRunway Statistics:")
print(df["runway_target"].describe())

# ==========================
# FEATURE ENGINEERING
# ==========================

df["liquidity_ratio"] = (
    df["totalCurrentAssets"] /
    df["totalCurrentLiabilities"]
)

df["debt_ratio"] = (
    df["totalLiab"] /
    df["totalAssets"]
)

df["profit_margin"] = (
    df["netIncome_x"] /
    df["totalRevenue"]
)

# Clean infinities

df = df.replace(
    [float("inf"), float("-inf")],
    0
)

# ==========================
# FEATURES
# ==========================

features = [
    "cash",
    "totalOperatingExpenses",
    "totalRevenue",
    "operatingIncome",
    "netIncome_x",
    "totalAssets",
    "totalLiab",
    "totalCurrentAssets",
    "totalCurrentLiabilities",
    "changeInCash",
    "longTermDebt",
    "netReceivables",
    "liquidity_ratio",
    "debt_ratio",
    "profit_margin"
]

# ==========================
# X AND y
# ==========================

X = df[features].fillna(0)

y = df["runway_target"]

# ==========================
# TRAIN TEST SPLIT
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# XGBOOST MODEL
# ==========================

model = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

# ==========================
# PREDICT
# ==========================

pred = model.predict(X_test)

# ==========================
# EVALUATION
# ==========================

print("\n=========================")
print("XGBOOST RESULTS")
print("=========================")

print(
    "R2 Score:",
    r2_score(y_test, pred)
)

print(
    "MAE:",
    mean_absolute_error(y_test, pred)
)

# ==========================
# FEATURE IMPORTANCE
# ==========================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop Features:")
print(importance.head(10))

# ==========================
# SAVE MODEL
# ==========================

import joblib

joblib.dump(
    model,
    "runway_model_xgb.pkl"
)

print("\nXGBoost Model Saved!")
