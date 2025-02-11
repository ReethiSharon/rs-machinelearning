import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Title
st.title('🤖 Breast Cancer Prediction App')

st.info("This app predicts whether a tumor is **Benign or Malignant** based on user inputs.")

# Load dataset
url = "https://raw.githubusercontent.com/ReethiSharon/rs-machinelearning/master/data.csv"
df = pd.read_csv(url)

# Convert diagnosis to numerical (M -> 1, B -> 0)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Split data into features and target
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model (Logistic Regression)
model = LogisticRegression()
model.fit(X_train, y_train)

# Sidebar User Inputs
with st.sidebar:
    st.header('🧬 Enter Tumor Features')

    # Get user inputs dynamically
    input_data = []
    for feature in X.columns:
        value = st.number_input(f"{feature}", float(df[feature].min()), float(df[feature].max()), float(df[feature].mean()))
        input_data.append(value)

    # ID input for patient identification
    patient_id = st.number_input("Enter Patient ID", min_value=int(df["id"].min()), max_value=int(df["id"].max()), step=1)
    
    # Predict button
    if st.button("Predict"):
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)  # Scale the input
        prediction = model.predict(input_scaled)[0]  # Get prediction
        probability = model.predict_proba(input_scaled)[0][1]  # Get probability of being malignant

        # Show the result with patient ID
        if prediction == 1:
            st.error(f"🔴 Patient ID: {patient_id} - The tumor is **Malignant (Cancerous)** (Confidence: {probability:.2%})")
        else:
            st.success(f"🟢 Patient ID: {patient_id} - The tumor is **Benign (Non-Cancerous)** (Confidence: {1 - probability:.2%})")
