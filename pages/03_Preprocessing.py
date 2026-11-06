from sklearn.experimental import enable_iterative_imputer
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from scipy import stats
from scipy.stats.mstats import winsorize

st.title("3- Data Preprocessing Page")

if 'data' not in st.session_state:
    st.warning("Please upload a dataset first!")
else:

    if 'process_data' not in st.session_state:
        st.session_state['process_data'] = st.session_state['data'].copy()
    
    process_data = st.session_state['process_data']
    st.markdown("Select Target Variable")
    target_col = st.selectbox(
        "Which column is your target?", 
        process_data.columns.tolist()
    )
    # نحفظه في الـ Station عشان كل الخطوات اللي تحت والصفحات التانية تشوفه
    st.session_state['target_col'] = target_col
    st.divider()

    if 'success_msgs' not in st.session_state:
        st.session_state['success_msgs'] = {}

    
    st.header("1. Clean Data")
    cols_to_drop = st.multiselect("Select columns to remove: ", process_data.columns.tolist())
    if st.button("Drop Selected Columns"):
        if cols_to_drop:
            process_data = process_data.drop(columns=cols_to_drop)
            st.session_state['process_data'] = process_data
            st.session_state['success_msgs']['drop'] = "Columns removed successfully!"
            st.rerun()
            
    if 'drop' in st.session_state['success_msgs']:
        st.success(st.session_state['success_msgs']['drop'])

    
    st.header("2. Handle Missing Values")
    num_cols_missing = process_data.select_dtypes(include=[np.number]).columns.tolist()
    impute_method = st.radio("Imputer Method:", ["None", "Mean (Simple)", "Median (Simple)", "IterativeImputer", "KNN Imputer"])
    
    if impute_method != "None" and st.button("Apply Imputation"):
        if impute_method == "Mean (Simple)":
            imputer = SimpleImputer(strategy='mean')
        elif impute_method == "Median (Simple)":
            imputer = SimpleImputer(strategy='median')
        elif impute_method == "IterativeImputer":
            imputer = IterativeImputer()
        else:
            imputer = KNNImputer(n_neighbors=5)
        
        process_data[num_cols_missing] = imputer.fit_transform(process_data[num_cols_missing])
        st.session_state['process_data'] = process_data
        st.session_state['success_msgs']['impute'] = f"Missing values handled using {impute_method}"
        st.rerun()

    if 'impute' in st.session_state['success_msgs']:
        st.success(st.session_state['success_msgs']['impute'])

    st.header("3. Outlier Handling")
    current_num_cols = process_data.select_dtypes(include=[np.number]).columns.tolist()
    outlier_cols = st.multiselect("Select column to clean outliers:", current_num_cols)
    method = st.radio("Select method to handiling outlier", ["IQR", "Z-Score", "winsorization"])
    
    if st.button("Apply Outlier Treatment"):
        if outlier_cols:
            if method == "IQR":
                for col in outlier_cols:
                    Q1 = process_data[col].quantile(0.25)
                    Q3 = process_data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    process_data[col] = np.clip(process_data[col], lower_bound, upper_bound)
            elif method == "Z-Score":
                for col in outlier_cols:
                    z_scores = np.abs(stats.zscore(process_data[col]))
                    process_data = process_data[z_scores < 3]
            elif method == "winsorization":
                for col in outlier_cols:
                    process_data[col] = winsorize(process_data[col], limits=[0.05, 0.05])
            
            st.session_state['process_data'] = process_data
            st.session_state['success_msgs']['outlier'] = f"{method} applied to selected columns!"
            st.rerun()
        else:
            st.warning("Please select at least one column.")

    if 'outlier' in st.session_state['success_msgs']:
        st.success(st.session_state['success_msgs']['outlier'])

    
    st.header("4. Data Normalization")
    scale_method = st.radio("Scaling Method:", ["None", "Standard Scaler", "MinMax Scaler"])

    if scale_method != "None":
        all_numeric_to_scale = process_data.select_dtypes(include=[np.number]).columns.tolist()
        target_col = st.session_state.get('target_col')
        if target_col in all_numeric_to_scale:
            all_numeric_to_scale.remove(target_col)
        if st.button("Apply Scaling"):
            if all_numeric_to_scale:
                
                scaler = StandardScaler() if scale_method == "Standard Scaler" else MinMaxScaler()
                process_data[all_numeric_to_scale] = scaler.fit_transform(process_data[all_numeric_to_scale])
                
                st.session_state['process_data'] = process_data
                st.session_state['scaler'] = scaler 
                st.session_state['success_msgs']['scale'] = f"Data scaled using {scale_method}"
                st.rerun()
            else:
                st.warning("No numeric columns available to scale.")

    if 'scale' in st.session_state['success_msgs']:
        st.success(st.session_state['success_msgs']['scale'])

   
    st.header("5. Data Encoding")
    cat_cols = process_data.select_dtypes(include=['object']).columns.tolist()
    if cat_cols:
        encode_method = st.radio("Encoding Type:", ["Label Encoder", "One-Hot Encoder"])
        cols_to_encode = st.multiselect("Select columns to encode:", cat_cols)
        
        if st.button("Apply Encoding"):
            if cols_to_encode:
                if encode_method == "Label Encoder":
                    le = LabelEncoder()
                    for col in cols_to_encode:
                        process_data[col] = le.fit_transform(process_data[col].astype(str))
                elif encode_method == "One-Hot Encoder":
                    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    encd_data = ohe.fit_transform(process_data[cols_to_encode]) 
                    encoded_df = pd.DataFrame(encd_data, columns=ohe.get_feature_names_out(cols_to_encode))
                    
                    process_data = pd.concat([process_data.drop(columns=cols_to_encode, errors="ignore").reset_index(drop=True), 
                                            encoded_df.reset_index(drop=True)], axis=1)
                    process_data = process_data.loc[:, ~process_data.columns.duplicated()]

                st.session_state['process_data'] = process_data
                st.session_state['success_msgs']['encode'] = f"{encode_method} Complete Successfully!"
                st.rerun()
            else:
                st.warning("Please select columns first.")
    else:
        st.info("No categorical columns left to encode.")

    if 'encode' in st.session_state['success_msgs']:
        st.success(st.session_state['success_msgs']['encode'])
    

    st.subheader("Final Processed Data Preview")
    st.dataframe(process_data.head())