import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the label encoders
with open('label_encoders.pkl', 'rb') as f:
    le_dict = pickle.load(f)

# App Configuration
st.set_page_config(page_title="💻 Laptop Price Predictor", layout="wide")
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
    }
    .title {
        color: #1261A0;
        text-align: center;
    }
    .subheader {
        color: #2D3142;
        font-size: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App Title
st.markdown("<h1 class='title'>💻 Laptop Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar - Input features
st.sidebar.header("🧾 Laptop Specifications")

brand = st.sidebar.selectbox("Brand", le_dict['Company'].classes_)
type_ = st.sidebar.selectbox("Laptop Type", le_dict['TypeName'].classes_)
ram = st.sidebar.selectbox("RAM (in GB)", [4, 8, 16, 32, 64])
weight = st.sidebar.slider("Weight (in kg)", 1.0, 5.0, 2.0, 0.1)
touchscreen = st.sidebar.radio("Touchscreen", ["No", "Yes"])
ips = st.sidebar.radio("IPS Display", ["No", "Yes"])
screen_size = st.sidebar.slider("Screen Size (in inches)", 11.0, 18.0, 15.6, 0.1)
resolution = st.sidebar.selectbox("Screen Resolution", [
    '1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800'])

cpu = st.sidebar.selectbox("CPU Brand", le_dict['Cpu brand'].classes_)
hdd = st.sidebar.selectbox("HDD (in GB)", [0, 128, 256, 512, 1024, 2048])
primary_storage_type = st.sidebar.selectbox("Primary Storage Type", le_dict['PrimaryStorageType'].classes_)
ssd = st.sidebar.selectbox("SSD (in GB)", [0, 128, 256, 512, 1024])
secondary_storage_type = st.sidebar.selectbox("Secondary Storage Type", le_dict['SecondaryStorageType'].classes_)
gpu = st.sidebar.selectbox("GPU Brand", le_dict['Gpu brand'].classes_)
os = st.sidebar.selectbox("Operating System", le_dict['os'].classes_)

# Feature Engineering
touchscreen = 1 if touchscreen == 'Yes' else 0
ips = 1 if ips == 'Yes' else 0

# Resolution handling
X_res, Y_res = map(int, resolution.split('x'))
ppi = ((X_res ** 2 + Y_res ** 2) ** 0.5) / screen_size

# Encoding using label encoders
brand_encoded = le_dict['Company'].transform([brand])[0]
type_encoded = le_dict['TypeName'].transform([type_])[0]
cpu_encoded = le_dict['Cpu brand'].transform([cpu])[0]
gpu_encoded = le_dict['Gpu brand'].transform([gpu])[0]
os_encoded = le_dict['os'].transform([os])[0]
primary_storage_type_encoded = le_dict['PrimaryStorageType'].transform([primary_storage_type])[0]
secondary_storage_type_encoded = le_dict['SecondaryStorageType'].transform([secondary_storage_type])[0]

# Prepare the final feature array in correct order
features = np.array([[brand_encoded, type_encoded, ram, weight,
                      touchscreen, ips, ppi, cpu_encoded,
                      hdd, primary_storage_type_encoded,
                      ssd, secondary_storage_type_encoded,
                      gpu_encoded, os_encoded]])

# Prediction
if st.button("Predict Price"):
    try:
        predicted_price = model.predict(features)[0]
        st.success(f"Estimated Laptop Price: ₹ {int(np.exp(predicted_price)):,}")
    except Exception as e:
        st.error(f"Oops! Something went wrong with prediction:\n{e}")
