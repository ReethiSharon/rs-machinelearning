import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Title
st.title('🩺 Breast Cancer Prediction App')

st.info("This app predicts whether a tumor is **Benign or Malignant** based on user inputs.")

# Load dataset
df = pd.read_csv(r"https://raw.githubusercontent.com/ReethiSharon/rs-machinelearning/master/data.csv")

# Convert diagnosis to numerical (M -> 1, B -> 0)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Keep 'id' for patient identification but not for prediction
patient_ids = df['id']  # Store patient IDs separately
df = df.drop(columns=['id'])  # Remove 'id' from features

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

# Compute feature importance
feature_importance = np.abs(model.coef_[0])
feature_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
feature_df = feature_df.sort_values(by="Importance", ascending=False)

# Sidebar User Inputs
with st.sidebar:
    st.header('🆔 Patient Identification')
    patient_id = st.number_input("Enter Patient ID", min_value=int(patient_ids.min()), max_value=int(patient_ids.max()), step=1)

    st.header('🧬 Enter Tumor Features')

    # Get user inputs dynamically
    input_data = []
    for feature in X.columns:
        value = st.number_input(
            f"{feature}",
            min_value=float(df[feature].min()),
            max_value=float(df[feature].max()),
            value=float(df[feature].mean()) if not np.isnan(df[feature].mean()) else 0.0
        )
        input_data.append(value)

    # Predict button
    if st.button("Predict"):
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array.astype(np.float64))  # Fix warning
        prediction = model.predict(input_scaled)[0]  # Get prediction
        probability = model.predict_proba(input_scaled)[0][1]  # Get probability of being malignant

        # Show the result
        st.subheader(f"🆔 Patient ID: **{patient_id}**")  # Display the entered ID
        if prediction == 1:
            st.error(f" 🔴The tumor is **Malignant (Cancerous)** (Confidence: {probability:.2%})")
        else:
            st.success(f"🟢The tumor is **Benign (Non-Cancerous)** (Confidence: {1 - probability:.2%})")

# Visualization: Top 10 Features
fig, ax = plt.subplots()
ax.barh(feature_df['Feature'][:10], feature_df['Importance'][:10], color='skyblue')
ax.set_xlabel("Importance Score")
ax.set_ylabel("Features")
ax.set_title("Top 10 Important Features")
plt.gca().invert_yaxis()  # Invert axis for better readability

# Display chart in Streamlit
st.pyplot(fig)
