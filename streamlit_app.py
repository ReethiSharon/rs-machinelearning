import streamlit as st
import pandas as pd
import numpy as np


st.title('🤖 Machine Learning App')

st.info('This app builds a machine learning model using breast cancer data.')

# Load the dataset
url = "https://raw.githubusercontent.com/ReethiSharon/rs-machinelearning/master/data.csv"
df = pd.read_csv(url)

with st.expander('Data'):
    st.write('**Raw Data**')
    st.dataframe(df)

    # Convert diagnosis column to numerical values
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

    st.write('**Processed Data**')
    st.dataframe(df)

    # Features (X) and Target (y)
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']

    st.write('**Features (X)**')
    st.dataframe(X)

    st.write('**Target (y)**')
    st.dataframe(y)

# Scatter plot visualization
with st.expander('Data Visualization'):
     st.write("**Scatter Plot: Mean Radius vs.Mean Perimeter**")
     st.scatter_chart(
         data=df,
         x='radius_mean',
         y = 'perimeter_mean',
         color='diagnosis'
         
     )
with st.sidebar:
   st.header('Input features')
   id=st.number_input("Enter your id.no",min_value=1,max_value=1000000000000)
    diagnosis = st.selectbox("Select Diagnosis", ["All", "Malignant", "Benign"])
    
    # Number Input for Custom Threshold (Fixed Column Name)
   
   radius_mean= st.slider("Select Mean Radius:", 
                       float(df["radius_mean"].min()), 
                       float(df["radius_mean"].max()), 
                       (df["radius_mean"].min(), df["radius_mean"].max()))
    
   area_range = st.slider("Select Area Mean Range:", 
                           float(df["area_mean"].min()), 
                           float(df["area_mean"].max()), 
                           (df["area_mean"].min(), df["area_mean"].max()))
    numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
    x_feature = st.selectbox("X-axis Feature", numeric_features)
    y_feature = st.selectbox("Y-axis Feature", numeric_features)
    
    # Checkbox for Normalization Option
     normalize_data=st.checkbox("Normalize Data?")
 if diagnosis != "All":
    df = df[df["diagnosis"] == diagnosis]

# Filter Data Based on Area Range
df = df[(df["area_mean"] >= area_range[0]) & (df["area_mean"] <= area_range[1])]

# Display Filtered Data with ID
st.write("### Filtered Data")
st.dataframe(df)

# Scatter Plot (Fixed Column Names)
st.write(f"### Scatter Plot: {x_feature} vs {y_feature}")
st.scatter_chart(data=df, x=x_feature, y=y_feature)
  

