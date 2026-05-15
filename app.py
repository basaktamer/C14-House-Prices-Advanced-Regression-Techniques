import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- 1. MODEL LOADING ---
def load_assets():
    # Corrected pickle logic: removed the .dump typo
    with open('lasso_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model_columns.pkl', 'rb') as f:
        columns = pickle.load(f)
    return model, columns

# Try to load the model assets
try:
    model, model_columns = load_assets()
except FileNotFoundError:
    st.error("Model files not found! Please ensure 'lasso_model.pkl' and 'model_columns.pkl' are in the app folder.")
    st.stop()

# --- 2. USER INTERFACE ---
st.set_page_config(page_title="House Price Predictor", page_icon="🏡")

st.title("🏡 House Price Predictor")
st.markdown("""
This app predicts the sale price of a house based on **engineered features**. 
Instead of 81 raw inputs, we use the **11 Super Variables** that drive the most value.
""")

st.divider()

# Organize inputs into two columns for better UI
col1, col2 = st.columns(2)

with col1:
    st.header("📍 Location & Quality")
    
    # Neighborhood selection (Cleaned up from OHE column names)
    neighborhoods = sorted([col.replace('Neighborhood_', '') for col in model_columns if 'Neighborhood_' in col])
    selected_neighborhood = st.selectbox("Select Neighborhood", neighborhoods)
    
    overall_score = st.slider("Overall Score (Qual + Cond + Functional)", 2, 21, 12, 
                              help="The combined score of house health and functionality.")
    
    exterior_score = st.slider("Exterior Quality/Condition Score", 2, 10, 6)
    
    is_remodeled = st.radio("Was the house remodeled?", ["Yes", "No"])
    is_remodeled_val = 1 if is_remodeled == "Yes" else 0

with col2:
    st.header("📐 Size & Age")
    
    total_sf = st.number_input("Total Square Footage (Basement + 1st + 2nd)", value=2000, step=50)
    
    total_bath = st.number_input("Total Bathrooms (Half baths count as 0.5)", value=2.0, step=0.5)
    
    house_age = st.number_input("Age of the House (Years)", value=10, min_value=0)
    
    lot_score = st.number_input("Lot Score (log1p of Lot Area)", value=9.0, format="%.2f",
                                help="Standard log value of the property area.")

st.divider()

with st.expander("More Features (Garage & Luxury)"):
    c1, c2 = st.columns(2)
    with c1:
        garage_score = st.number_input("Garage Score (Cars × Area)", value=800)
    with c2:
        luxury_score = st.number_input("Luxury Score (Fireplaces + Pool)", value=1)

# --- 3. PREDICTION LOGIC ---
if st.button("Calculate Predicted Sale Price", type="primary", use_container_width=True):
    
    # 1. Create a zero-filled dataframe matching the training features (195-196 cols)
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)
    
    # 2. Assign the Numerical Super Variables
    input_df['TotalSF'] = total_sf
    input_df['TotalBath'] = total_bath
    input_df['HouseAge'] = house_age
    input_df['IsRemodeled'] = is_remodeled_val
    input_df['GarageScore'] = garage_score
    input_df['LuxuryScore'] = luxury_score
    input_df['LotScore'] = lot_score
    input_df['ExteriorScore'] = exterior_score
    input_df['OverallScore'] = overall_score
    
    # Note: RemodAge and TotalAbvGrdRooms can be added here if included in your final model_columns
    if 'RemodAge' in model_columns:
        input_df['RemodAge'] = house_age # Simple assumption for the app
    
    # 3. Handle the Neighborhood (One-Hot Encoding alignment)
    target_neigh_col = f"Neighborhood_{selected_neighborhood}"
    if target_neigh_col in model_columns:
        input_df[target_neigh_col] = 1
    
    # 4. Perform Prediction
    # Since model was trained on log(SalePrice), we use expm1 to get dollars back
    log_prediction = model.predict(input_df)
    final_price = np.expm1(log_prediction)[0]
    
    # 5. Display Result
    st.balloons()
    st.success(f"### Estimated Market Value: ${final_price:,.2f}")
    
    st.info("""
    **Insight:** This prediction uses a LassoCV model. Location and Total Square Footage are 
    typically the strongest influencers in this specific prediction.
    """)