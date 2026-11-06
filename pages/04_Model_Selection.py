import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC,SVR
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.cluster import KMeans,DBSCAN
import time
st.title("4- Model Selection Page")

if 'data' not in st.session_state:
     st.warning("please upload data and prepocess it")
else:
    copyed_data = st.session_state['process_data'].copy()
    st.write("currrent data shape ",copyed_data.shape)

    st.header("select the model")
    model_name = st.radio("select model:",["None","linear Regrssion","logistic Regression",
                                            "Random forest classifier","Random forest regressor",
                                            "svm_classifier","svm_regression","KNN_regression","KNN_classifier",
                                            "Decision tree regresion","Decisison tree classifier","k_mean","DBSCAN"])
   

    target_col = st.session_state["target_col"]

    X = copyed_data.drop(target_col,axis = 1)
    Y = copyed_data[target_col]

    X_train,X_test,y_train,y_test = train_test_split(X,Y, test_size= .2, random_state= 42)
    st.session_state['X_test'] = X_test
    st.session_state['y_test'] = y_test
    model = None
     
     
    if model_name != "None":
        st.info(f"Training set: {X_train.shape[0]} samples | Testing set: {X_test.shape[0]} samples")
       
    
        if model_name =="linear Regrssion":
            model =LinearRegression()
        elif model_name == "logistic Regression":
            model = LogisticRegression()
        elif model_name =="Random forest classifier":
            model = RandomForestClassifier(max_depth= 5,random_state= 42,n_estimators= 50)
        elif model_name == "Random forest regressor":
            model = RandomForestRegressor(max_depth=5,random_state=42,n_estimators= 50)
        elif model_name == "svm_classifier":
            model = SVC(kernel='rbf', C=1.0)
        elif model_name == "svm_regression":
            model = SVR(kernel='rbf', C=1.0)
        elif model_name == "KNN_regression":
            model = KNeighborsRegressor(n_neighbors= 5)
        elif model_name == "KNN_classifier":
            model = KNeighborsClassifier(n_neighbors=5)
        elif model_name == "Decision tree regresion":
            model = DecisionTreeRegressor(max_depth=5)
        elif model_name == "Decisison tree classifier":
            model = DecisionTreeClassifier(max_depth=5)
        elif model_name == "k_mean":
            model = KMeans(n_clusters= 5)
        elif model_name == "DBSCAN":
            model = DBSCAN()

    start_time = time.time()
    if model is not None:
            with st.spinner(f"Training {model_name}..."):
                if model_name == "k_mean" or model_name == "DBSCAN":
                    model.fit(X) 

                else:
                    model.fit(X_train, y_train)
            end_time = time.time()
            duration = round(end_time - start_time, 2)
            st.session_state["trained_model"] = model  
            st.session_state["model_name"] = model_name 
            st.success(f"{model_name} trained in {duration} seconds!")

            
          


     
