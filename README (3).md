# 📊 Customer Lifetime Value (CLV) Prediction Dashboard

A Streamlit web application that predicts the **Customer Lifetime Value** based on customer transaction history using machine learning (Random Forest Regressor).

---

## 🚀 Features

- Uploads and preprocesses transactional customer data
- Calculates CLV-relevant features like Recency, Frequency, and Monetary Value
- Predicts future Customer Lifetime Value using a trained Random Forest model
- Visualizes CLV predictions with a histogram
- Displays model performance metrics (MAE, RMSE, R²)
- Fully interactive UI built using Streamlit

---

## 🧪 Machine Learning Details

- **Model:** Random Forest Regressor
- **Features Used:**
  - Recency (days since last purchase)
  - Total Purchases
  - Average Spent per Purchase
- **Target:** Total Spent (log-transformed CLV)
- **Data Handling:**
  - Outlier Winsorization
  - Log Transformation
  - Feature Scaling using `StandardScaler`

---

## 📁 Dataset

> Make sure the Excel file (`customer.xlsx`) is placed correctly or update the path in the script:
```python
df = pd.read_excel(r"C:\Users\addya\Downloads\customer.xlsx")
```

You can replace the path with a relative path or use file upload functionality for deployment.

---

## ▶️ How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the App
```bash
streamlit run clv.py
```

---

## 📊 Output

- Predicts CLV based on user input
- Histogram plot of predicted CLVs
- Real-time evaluation metrics displayed on the dashboard

---

## 🧠 Sample Prediction Inputs (Sidebar)
- Recency: 30
- Total Purchases: 10
- Avg Spent: $50.00

---

## 📦 Files Included
- `clv.py`: Main Streamlit app with data handling, model training, and prediction
- `clv_model.pkl`: Trained model (auto-saved)
- `scaler.pkl`: Trained StandardScaler object (auto-saved)
- `requirements.txt`: Python dependencies
- `README.md`: Documentation file

---

## 📌 Notes

- Ensure Excel file is clean and has columns: `CustomerID`, `Quantity`, `UnitPrice`, `InvoiceDate`
- This is a beginner-friendly ML app—perfect for data science portfolios!

---

## ✨ Author

**Adham Ansari**  
Machine Learning | Data Science | Web Scraping | Python Automation
