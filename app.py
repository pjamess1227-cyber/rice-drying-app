import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.title("Rice Drying Prediction Model")

# ================================
# LOAD DATASET
# ================================
@st.cache_data
def load_data():
    return pd.read_excel("optimized_drying_dataset_4to6hrs_300rows.xlsx")

df = load_data()

X = df[["Time (min)", "Temperature (°C)", "Humidity (%)"]]
y = df["Moisture Content (%)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

# ================================
# TEMPERATURE SIMULATION
# ================================
def simulate_temp(time):
    if time <= 60:
        return 22 + (30/60) * time  # rise to ~52°C
    else:
        return np.random.uniform(50, 55)

# ================================
# PREDICTION FUNCTION
# ================================
def predict_drying_time(target_mc=14):
    time = 0
    steps = 0
    log = []

    while True:
        temp = simulate_temp(time)
        humidity = np.random.uniform(55, 65)

        input_df = pd.DataFrame([[time, temp, humidity]],
                                columns=["Time (min)", "Temperature (°C)", "Humidity (%)"])

        pred_mc = model.predict(input_df)[0]

        log.append((time, temp, humidity, pred_mc))
        steps += 1

        if pred_mc <= target_mc:
            return time, steps, log

        time += 1

        if time > 500:
            return None, steps, log

# ================================
# RUN BUTTON
# ================================
if st.button("Run Prediction"):

    predicted_time_min, total_steps, log = predict_drying_time()
    predicted_time_hr = predicted_time_min / 60

    initial_weight = 3.0
    final_weight = np.random.uniform(2.6, 2.7)
    moisture_loss = initial_weight - final_weight

    power_kw = 1.05
    energy = power_kw * predicted_time_hr
    efficiency = moisture_loss / energy

    st.subheader("Results")
    st.write(f"Predicted Drying Time: {predicted_time_hr:.2f} hours")
    st.write(f"Total Iterations: {total_steps}")
    st.write(f"Initial Weight: {initial_weight:.2f} kg")
    st.write(f"Final Weight: {final_weight:.2f} kg")
    st.write(f"Moisture Loss: {moisture_loss:.2f} kg")
    st.write(f"Energy Efficiency: {efficiency:.3f} kg/kWh")

    st.subheader("Prediction Process")
    log_df = pd.DataFrame(log, columns=[
        "Time (min)", "Temperature (°C)", "Humidity (%)", "Predicted MC (%)"
    ])
    st.dataframe(log_df)