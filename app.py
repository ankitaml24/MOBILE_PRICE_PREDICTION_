
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Mobile Price Predictor",
    page_icon="📱",
    layout="wide"
)

# Load trained model
data = joblib.load("mobile_price_model.pkl")
model = data["model"]

st.title("📱 Mobile Phone Price Category Prediction")
st.write("Enter mobile specifications to predict its price category.")
st.divider()

st.subheader("Enter Mobile Specifications")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        battery = st.slider("Battery Power (mAh)", 500, 2000, 1500)
        ram = st.slider("RAM (MB)", 256, 4000, 2000)
        memory = st.slider("Internal Memory (GB)", 2, 64, 32)
        weight = st.slider("Mobile Weight (g)", 80, 200, 150)
        clock = st.slider("Clock Speed (GHz)", 0.5, 3.0, 1.5)
        cores = st.selectbox("Processor Cores", list(range(1, 9)), index=3)
        front_camera = st.slider("Front Camera (MP)", 0, 20, 8)
        back_camera = st.slider("Rear Camera (MP)", 0, 20, 12)
        talk = st.slider("Talk Time (hours)", 2, 20, 10)

    with col2:
        px_height = st.slider("Pixel Height", 0, 1960, 800)
        px_width = st.slider("Pixel Width", 500, 2000, 1200)
        screen_h = st.slider("Screen Height (cm)", 5, 20, 12)
        screen_w = st.slider("Screen Width (cm)", 0, 18, 7)
        depth = st.slider("Phone Depth (cm)", 0.1, 1.0, 0.5)
        bluetooth = st.selectbox("Bluetooth", [0, 1])
        dual_sim = st.selectbox("Dual SIM", [0, 1])
        four_g = st.selectbox("4G Support", [0, 1])
        three_g = st.selectbox("3G Support", [0, 1])
        touch = st.selectbox("Touch Screen", [0, 1])
        wifi = st.selectbox("Wi-Fi", [0, 1])

    submitted = st.form_submit_button("Predict Price Category")

if submitted:
    features = {
        "battery_power": battery,
        "blue": bluetooth,
        "clock_speed": clock,
        "dual_sim": dual_sim,
        "fc": front_camera,
        "four_g": four_g,
        "int_memory": memory,
        "m_dep": depth,
        "mobile_wt": weight,
        "n_cores": cores,
        "pc": back_camera,
        "px_height": px_height,
        "px_width": px_width,
        "ram": ram,
        "sc_h": screen_h,
        "sc_w": screen_w,
        "talk_time": talk,
        "three_g": three_g,
        "touch_screen": touch,
        "wifi": wifi
    }

    input_df = pd.DataFrame([features])[data["features"]]
    prediction = int(model.predict(input_df)[0])

    categories = {
        0: "Low Cost",
        1: "Medium Cost",
        2: "High Cost",
        3: "Very High Cost"
    }

    st.success("Prediction completed!")
    st.subheader("Predicted Category: " + categories[prediction])

    probabilities = model.predict_proba(input_df)[0]
    chart = pd.DataFrame({
        "Category": [
            "Low Cost", "Medium Cost",
            "High Cost", "Very High Cost"
        ],
        "Probability (%)": probabilities * 100
    }).set_index("Category")

    st.write("Prediction probability by category:")
    st.bar_chart(chart)