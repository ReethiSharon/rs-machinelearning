import streamlit as st
import pandas as pd


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
    st.scatter_chart(data=df,x='mean_radius',y = 'mean_perimeter',color='diagnosis')
    

