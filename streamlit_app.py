import streamlit as st
import pandas as pd

st.title('🤖 Machine Learning App')

st.info('This app builds a machine learning model using breast cancer data.')

# Load the dataset
url = "https://raw.githubusercontent.com/ReethiSharon/rs-machinelearning/master/data.csv"
df = pd.read_csv(url)

# Convert diagnosis to numerical (M -> 1, B -> 0)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

with st.expander('Data'):
    st.write('**Raw Data**')
    st.dataframe(df)

    st.write('**Processed Data**')
    st.dataframe(df)

    # Features (X) and Target (y)
    X = df.drop(['id', 'diagnosis'], axis=1)  # Drop ID from features
    y = df['diagnosis']

    st.write('**Features (X)**')
    st.dataframe(X)

    st.write('**Target (y)**')
    st.dataframe(y)

# Sidebar Inputs
with st.sidebar:
    st.header('🔍 Input Features')

    id_input = st.number_input("Enter your ID", min_value=int(df["id"].min()), max_value=int(df["id"].max()), step=1)

    diagnosis = st.selectbox("Select Diagnosis", ["All", "Malignant", "Benign"])

    radius_mean = st.slider("Select Mean Radius:", 
                            float(df["radius_mean"].min()), 
                            float(df["radius_mean"].max()), 
                            (df["radius_mean"].min()))

    area_range = st.slider("Select Area Mean Range:", 
                           float(df["area_mean"].min()), 
                           float(df["area_mean"].max()), 
                           (df["area_mean"].min(), df["area_mean"].max()))

    numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
    x_feature = st.selectbox("X-axis Feature", numeric_features)
    y_feature = st.selectbox("Y-axis Feature", numeric_features)

    normalize_data = st.checkbox("Normalize Data?")

# Filter Data Based on Diagnosis Selection
if diagnosis != "All":
    diagnosis_value = 1 if diagnosis == "Malignant" else 0
    df = df[df["diagnosis"] == diagnosis_value]

# Filter Data Based on Area Range
df = df[(df["area_mean"] >= area_range[0]) & (df["area_mean"] <= area_range[1])]

# Display Filtered Data
st.write("### Filtered Data")
st.dataframe(df)

# Convert diagnosis back to categorical for coloring
df["diagnosis_category"] = df["diagnosis"].map({1: "Malignant", 0: "Benign"})

# Scatter Plot
st.write(f"### Scatter Plot: {x_feature} vs {y_feature}")
st.scatter_chart(data=df, x=x_feature, y=y_feature, color="diagnosis_category")
