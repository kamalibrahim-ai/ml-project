import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("2- Data Visualization Page")


if 'data' not in st.session_state:
    st.warning("Please upload a dataset in the first page to see visualizations.")
else:
    df = st.session_state['data']
    
    st.header("Visualize Data") 
    
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    chart_type = st.selectbox("Choose Chart Type:", ["Line Plot", "Scatter Plot", "Box Plot"]) # [cite: 7]


    fig, ax = plt.subplots(figsize=(10, 6))

    if chart_type == "Line Plot":
        if len(numeric_cols) >= 1:
            x_axis = st.radio("Select X axis:", numeric_cols)
            y_axis = st.radio("Select Y axis:", numeric_cols)
            sns.lineplot(data=df, x=x_axis, y=y_axis, ax=ax)
            st.pyplot(fig)
        else:
            st.error("Line plot requires numerical columns.")

    elif chart_type == "Scatter Plot":
        if len(numeric_cols) >= 2:
            x_axis = st.radio("Select X axis:", numeric_cols)
            y_axis = st.radio("Select Y axis:", numeric_cols)
            sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax)
            st.pyplot(fig)
        else:
            st.error("Scatter plot requires at least 2 numerical columns.")

    elif chart_type == "Box Plot":
        if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
            x_axis = st.radio("Select Categorical column (X):", categorical_cols)
            y_axis = st.radio("Select Numerical column (Y):", numeric_cols)
            sns.boxplot(data=df, x=x_axis, y=y_axis, ax=ax)
            st.pyplot(fig)
        else:
            st.error("Box plot requires one categorical and one numerical column.")