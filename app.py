import streamlit as st
import pandas as pd

st.set_page_config(page_title="ML Project", layout="wide")

st.title("1- File Upload Page")

uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=['csv', 'xlsx'])

if uploaded_file is not None:

    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    st.success("Confirmation: File Uploaded Successfully!")
    
    st.subheader("Data Preview")
    st.dataframe(df.head())
    
    st.session_state['data'] = df
else:
    st.info("Please upload a file to start.")