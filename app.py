import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

# --- Streamlit Page Config ---
st.set_page_config(page_title="Machine Learning Regression App", layout="wide")

st.title("Interactive Machine Learning Regression App")
# App description
st.write("""\
# 🧠 Interactive Machine Learning Regression App\n \
\n \
Welcome to the **ML Regression Playground**!  \n \
This app allows you to **upload your CSV dataset** (with the **first row as headers**)  \n \
and explore data, preprocess it, and train multiple regression models — all interactively.\n \
\
---\
""")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Preview", "Info" , "📊 Summary", "📈 Plots"])
    with tab1:
        st.dataframe(df)
    with tab2:
        st.write(df.info())
    with tab3:
        st.write(df.describe().T)
    

    # --- Select Inputs and Output ---
    with st.expander("⚙️ Select Model Inputs"):
        target_col = st.selectbox("Select the Target (Output) Column", df.columns)
        input_cols = st.multiselect(
            "Select the Input (Feature) Columns", 
            [col for col in df.columns if col != target_col]
        )

    if input_cols and target_col:
        X = df[input_cols]
        y = df[target_col]

        # --- Split Data ---
        test_size = st.slider("Test Data Size (%)", 10, 50, 20) / 100
        random_state = st.number_input("Random State (for reproducibility)", 0, 1000, 42)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # --- Choose Algorithm(s) ---
        st.subheader("Select Regression Algorithms")
        algorithms = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Lasso Regression": Lasso(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "Support Vector Regressor": SVR(),
        }

        selected_models = st.multiselect(
            "Choose algorithms to train", list(algorithms.keys()), default=["Linear Regression"]
        )

        results = []

        # --- Train & Evaluate Models ---
        if st.button("🚀 Train and Evaluate"):
            st.write("### Results")

            for model_name in selected_models:
                model = algorithms[model_name]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                results.append({
                    "Model": model_name,
                    "R² Score": round(r2, 4),
                    "MAE": round(mae, 4),
                    "RMSE": round(rmse, 4),
                })

            results_df = pd.DataFrame(results).set_index("Model")
            st.dataframe(results_df.style.highlight_max(axis=0, color="lightgreen"))

            # --- Optional: Show Predictions ---
            with st.expander("📈 Show Predictions Comparison"):
                comp_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
                st.line_chart(comp_df)

else:
    st.info("👆 Please upload a CSV file to start.")
