# Demand Forecasting using Machine Learning (Scikit-learn)

This project uses a machine learning approach to forecast product demand in a supply chain setting. By analyzing historical data and extracting useful patterns, the model can help businesses make informed decisions about inventory, planning, and operations.

---

## Project Objective

To build a machine learning model that predicts the **number of products sold**, based on historical data and other influencing factors such as time-based features and categorical variables.

---

## Tools and Technologies

- **Python**
- **Pandas, NumPy** – Data handling
- **Matplotlib, Seaborn** – Data visualization
- **Scikit-learn** – Preprocessing, model building, and evaluation
- **RandomForestRegressor** – Prediction model
- **Joblib** – Model serialization (saving/loading)

---

## Dataset Overview

The dataset (`supply_chain_data.csv`) contains:
- Product details
- Sales history (`Number of products sold`)
- External factors (e.g., Promotion, Weather, Economic Indicators)
- A synthetic `Date` column added to extract time-based features like `Month`, `Quarter`, and `DayOfWeek`.

---

## Steps Performed

### 1. **Data Preprocessing**
- Dropped missing values
- Added synthetic daily dates
- Extracted `Month`, `DayOfWeek`, and `Quarter`
- One-hot encoded categorical features
- Standardized features using `StandardScaler`

### 2. **Model Building**
- Used `RandomForestRegressor` from Scikit-learn
- Split data into training and testing sets
- Trained the model on scaled data

### 3. **Model Evaluation**
- Evaluated using **Mean Squared Error (MSE)**
- Plotted predicted vs actual sales

### 4. **Model Deployment**
- Saved trained model and scaler using `joblib`
- Loaded model to predict new sales data

---

## Sample Prediction

```python
# Example new data (after encoding and scaling)
new_data = np.array([[...]])
new_data_scaled = scaler.transform(new_data)
predicted_sales = model.predict(new_data_scaled)

