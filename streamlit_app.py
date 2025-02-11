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
   user_id=st.number_input("Enter your id.no",min_value=1,max_value=100,step=1)
     
   diagnosis=st.selectbox('Diagnosis',('Malignant','Benign'))

