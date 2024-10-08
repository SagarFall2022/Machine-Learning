import streamlit as st
from model import train_model, predict_performance, load_model, generate_maintenance_insights

# Train the model if it doesn't exist
try:
    load_model()
except FileNotFoundError:
    train_model()

st.title('Truck Suspension Performance Predictor')

suspension_travel = st.number_input('Suspension Travel (mm)', min_value=10, max_value=30)
load_weight = st.number_input('Load Weight (kg)', min_value=1000, max_value=3000)
shock_pressure = st.number_input('Shock Absorber Pressure (psi)', min_value=100, max_value=400)
vibration_intensity = st.number_input('Vibration Intensity (g)', min_value=0.1, max_value=2.0)
temperature = st.number_input('Ambient Temperature (°F)', min_value=50, max_value=100)
mileage = st.number_input('Mileage (km)', min_value=10000, max_value=200000)

if st.button('Predict Performance'):
    prediction = predict_performance(suspension_travel, load_weight, shock_pressure, vibration_intensity, temperature, mileage)
    st.write(f'Predicted Performance Metric: {prediction:.2f}')

    insights = generate_maintenance_insights(prediction)
    st.write("Maintenance Insights:")
    st.write(insights)
