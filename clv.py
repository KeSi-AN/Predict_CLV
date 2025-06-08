import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats.mstats import winsorize

# Load Data (Modify as per your dataset location)
df = pd.read_excel(r"C:\Users\addya\Downloads\customer.xlsx")

# Handle Missing Values
df.dropna(subset=['CustomerID'], inplace=True)

# Remove Negative & Zero Quantities
df = df[df["Quantity"] > 0]
df = df[df["UnitPrice"] > 0]

# Compute Total Monetary Value per transaction
df["TotalValue"] = df["Quantity"] * df["UnitPrice"]

# Aggregate Customer-Level Data
clv_df = df.groupby("CustomerID").agg({
    "InvoiceDate": ["min", "max", "count"],  # Recency & Frequency
    "TotalValue": ["sum", "mean"]  # Monetary Value
}).reset_index()

clv_df.columns = ["CustomerID", "FirstPurchase", "LastPurchase", "TotalPurchases", "TotalSpent", "AvgSpent"]

# Convert Dates
clv_df["FirstPurchase"] = pd.to_datetime(clv_df["FirstPurchase"])
clv_df["LastPurchase"] = pd.to_datetime(clv_df["LastPurchase"])

# Compute Recency (Days since last purchase)
clv_df["Recency"] = (clv_df["LastPurchase"].max() - clv_df["LastPurchase"]).dt.days

# Handle Outliers (Winsorization)
clv_df["TotalSpent"] = winsorize(clv_df["TotalSpent"], limits=[0.05, 0.05])
clv_df["AvgSpent"] = winsorize(clv_df["AvgSpent"], limits=[0.05, 0.05])

# Log Transform Monetary Features to Handle Skewness
clv_df["TotalSpent"] = np.log1p(clv_df["TotalSpent"])
clv_df["AvgSpent"] = np.log1p(clv_df["AvgSpent"])

# Features & Target Variable
X = clv_df[["Recency", "TotalPurchases", "AvgSpent"]]
y = clv_df["TotalSpent"]  # Predicting future CLV

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Model (Random Forest)
model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10)
model.fit(X_train, y_train)

# Save Model & Scaler
joblib.dump(model, "clv_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Predictions
y_pred = model.predict(X_test)

# Evaluate Model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Streamlit App
st.title("Customer Lifetime Value (CLV) Prediction Dashboard")
st.sidebar.header("Enter Customer Details")
recency = st.sidebar.number_input("Recency (days since last purchase)", min_value=0, max_value=1000, value=30)
total_purchases = st.sidebar.number_input("Total Purchases", min_value=1, max_value=500, value=10)
avg_spent = st.sidebar.number_input("Average Spent per Purchase ($)", min_value=0.1, max_value=10000.0, value=50.0)

if st.sidebar.button("Predict CLV"):
    input_data = np.array([[recency, total_purchases, np.log1p(avg_spent)]])
    input_scaled = scaler.transform(input_data)
    predicted_clv = model.predict(input_scaled)[0]
    predicted_clv = np.expm1(predicted_clv)
    st.subheader(f"Predicted Customer Lifetime Value: ${predicted_clv:.2f}")

st.subheader("CLV Distribution")
fig, ax = plt.subplots()
sns.histplot(y_pred, kde=True, color='skyblue', ax=ax)
ax.set_xlabel("CLV")
ax.set_ylabel("Frequency")
st.pyplot(fig)

st.markdown("---")
st.write(f"Model Performance: MAE = {mae:.4f}, RMSE = {rmse:.4f}, R² = {r2:.4f}")
