---
title: House Price Predictor - Lasso Regression
emoji: 🏠
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.32.0
python_version: '3.10'
app_file: app.py
pinned: false
license: mit
---

# House Price Prediction using Lasso Regression

## 📌 Project Overview
This project predicts residential house prices based on the Ames Housing Dataset. Instead of using raw, noisy data, this model utilizes **Feature Engineering** to combine 81 columns into 11 high-impact "Super Variables."

## 🚀 Features
- **TotalSF:** Combined square footage of basement and above-ground levels.
- **OverallScore:** A unified metric of Quality, Condition, and Functionality.
- **HouseAge:** Calculated from year sold and year built.
- **Model:** Lasso Regression (L1 Regularization) for automated feature selection.

## 🛠️ Technical Setup
- **Language:** Python 3.10
- **Libraries:** Scikit-Learn, Pandas, Numpy, Streamlit
- **Deployment:** Hugging Face Spaces

## 📊 Key Insights
The model achieved an **RMSE of 0.14420** on the log-transformed Sale Price. The most significant predictors identified were neighborhood location, total bathrooms, and the overall quality score.

## 📂 Repository Structure
- `app.py`: Streamlit application script.
- `lasso_model.pkl`: Trained LassoCV model.
- `model_columns.pkl`: List of feature names for input alignment.
- `requirements.txt`: Environment dependencies.