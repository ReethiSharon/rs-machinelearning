import streamlit as st
import pandas as pd

st.title('🤖Machine Learning App')

st.info('This is app building machine learning model')

with st.expander('Data'):
   st.write('**Raw data**')
   df=pd.read_csv('https://raw.githubusercontent.com/ReethiSharon/rs-machinelearning/refs/heads/master/data.csv')
   df

   st.write('**X**')
   X=df.drop('diagnosis',axis=1)
   X

  st.write('**y**')
  y=df.diagnosis
  y











