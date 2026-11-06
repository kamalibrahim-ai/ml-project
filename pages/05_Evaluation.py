import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

st.title("5- Model Evaluation Page")
# st.write(st.session_state.keys())

classification_models = [
    "logistic Regression", 
    "Random forest classifier", 
    "svm_classifier", 
    "KNN_classifier", 
    "Decisison tree classifier"
]

regression_models = [
    "linear Regrssion", 
    "Random forest regressor", 
    "svm_regression", 
    "KNN_regression", 
    "Decision tree regresion"
]

clustering_models = [
    "k_mean", 
    "DBSCAN"
]

if 'trained_model' not in st.session_state or 'model_name' not in st.session_state:
    st.warning("Please train a model first.")
elif 'X_test' not in st.session_state or 'y_test' not in st.session_state:
    st.warning("Test data not foundز")
else:
    model = st.session_state['trained_model']
    model_name = st.session_state['model_name']
    X_test = st.session_state['X_test']
    y_test = st.session_state['y_test']
    target_col = st.session_state["target_col"]

    st.subheader(f"Results for: {model_name}")

    
    if model_name in classification_models:
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        
        st.metric("Test Accuracy", f"{acc:.2%}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("Confusion Matrix:")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            st.pyplot(fig)
            
        with col2:
            st.markdown("Classification Report:")
            report = classification_report(y_test, y_pred, output_dict=True)
            df_report = pd.DataFrame(report).transpose()
            st.dataframe(df_report.style.format("{:.2f}"))

    
    elif model_name in regression_models:
        
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("r2_score", f"{r2:.4f}")
        col2.metric("Mean Squared Error", f"{mse:.2f}")
        col3.metric("Mean Absolute Error", f"{mae:.2f}")
        
        st.markdown("Actual vs Predicted Values:")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.5, color='blue')
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color='red')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        st.pyplot(fig)

   
    elif model_name in clustering_models:
        
        try:
            from sklearn.metrics import silhouette_score
            if 'X' in st.session_state:
                full_data = st.session_state['X']
            else:
                full_data = st.session_state['process_data']
                full_data = full_data.drop(target_col,axis = 1)

            if model_name == "k_mean":
                labels = model.predict(full_data)
                
            elif model_name == "DBSCAN":
                 labels = model.labels_

            unique_labels = set(labels) - {-1} 
            
            if len(unique_labels) >= 2:
                score = silhouette_score(full_data, labels)
                st.metric("Silhouette Score (Full Data)", f"{score:.4f}")
                
                st.markdown("**Clusters Distribution:**")
                fig, ax = plt.subplots()
                scatter = ax.scatter(full_data.iloc[:, 0], full_data.iloc[:, 5], c=labels, cmap='viridis', alpha=0.6)
                plt.xlabel(full_data.columns[0])
                plt.ylabel(full_data.columns[1])
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Error details: {e}")
            
    else:
        st.error("Invalid model selected or model not recognized.")

    