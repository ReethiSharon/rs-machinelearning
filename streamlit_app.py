import streamlit as st
import pandas as pd
import plotly.express as px

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
    st.write('**Scatter Plot: Mean Radius vs. Mean Perimeter**')

    fig = px.scatter(df, x="mean radius", y="mean perimeter", 
                     color=df["diagnosis"].map({1: "Malignant", 0: "Benign"}), 
                     title="Mean Radius vs. Mean Perimeter (Colored by Diagnosis)")
    st.plotly_chart(fig)
