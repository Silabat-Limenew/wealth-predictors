
import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title='Household Wealth Prediction Dashboard', layout='wide')

st.title('🏠 Household Wealth Index Predictor')
st.write('Predict and explore household wealth index categories using our trained LightGBM model.')

# Load model and encoder
@st.cache_resource
def load_artifacts():
    model = joblib.load('lightgbm_wealth_model.joblib')
    encoder = joblib.load('target_encoder.joblib')
    return model, encoder

try:
    model, encoder = load_artifacts()
    st.success('Model loaded successfully!')
except Exception as e:
    st.error(f'Error loading model: {e}')
    st.stop()

# User Inputs
st.sidebar.header('Input Household Features')
st.write('### Make a Prediction')

# Let's create an input form using some of the top 5 features
urban_rural = st.selectbox('Type of Place of Residence (UrbanRural)', ['Rural', 'Urban'])
hv201 = st.selectbox('Source of Drinking Water (hv201)', ['Piped water', 'Well', 'Surface water', 'Other'])
hv205 = st.selectbox('Type of Toilet Facility (hv205)', ['No facility/bush/field', 'Pit latrine', 'Flush toilet', 'Other'])
hv271a = st.number_input('Wealth Index Factor Score (hv271a)', value=0.0)

# Run prediction
if st.button('Predict Wealth Index'):
    # Prepare a dummy dataframe replicating the 348 trained feature columns
    # For demo, we fill with defaults and update user-provided values
    feature_cols = model.feature_name_
    input_data = pd.DataFrame(0.0, index=[0], columns=feature_cols)
    
    # Update known top features mapped back to model encoders
    input_data['UrbanRural'] = 1 if urban_rural == 'Urban' else 0
    input_data['hv271a'] = hv271a
    
    # Run prediction probabilities
    probabilities = model.predict_proba(input_data)[0]
    predicted_class_idx = np.argmax(probabilities)
    predicted_class = encoder.classes_[predicted_class_idx]
    
    st.write(f'### Predicted Class: **{predicted_class.upper()}**')
    
    # Probability Chart
    prob_df = pd.DataFrame({
        'Wealth Class': encoder.classes_,
        'Probability': probabilities
    })
    st.bar_chart(data=prob_df, x='Wealth Class', y='Probability')
