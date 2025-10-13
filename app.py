# app.py
import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pickle
from scipy import stats

st.set_page_config(page_title="Interactive ML Regression App", layout="wide")


# --- Helper functions ---
def init_session_state():
    if "df" not in st.session_state:
        st.session_state.df = None
    if "models" not in st.session_state:
        st.session_state.models = {}  # key -> {model, metrics, params, timestamp}
    if "split" not in st.session_state:
        st.session_state.split = {}
    if "last_train" not in st.session_state:
        st.session_state.last_train = None


params = {
    "xtick.top": True,
    "ytick.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
}
plt.rcParams.update(params)


def load_csv(uploaded_file):
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Couldn't read CSV: {e}")
        return None


# @st.cache_data
def describe(df: pd.DataFrame):
    stats = df.describe().T
    stats["skewness"] = df.skew()
    stats["kurtosis"] = df.kurtosis()
    return stats


def customize_plot(df, xdata, ydata, scatter_kwargs=None, ax_kwargs=None):
    fig, ax = plt.subplots()

    ax.scatter(df[xdata].values, df[ydata], **scatter_kwargs)

    if ax_kwargs:
        for k, v in ax_kwargs.items():
            setter = getattr(ax, f"set_{k}", None)
            if callable(setter):
                setter(v)

    return fig


def build_pipeline(model_obj, use_scaler=True):
    steps = []
    # Impute numeric with median
    steps.append(("imputer", SimpleImputer(strategy="median")))
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model_obj))
    return Pipeline(steps)


def compute_metrics(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def plot_predictions(y_true, y_pred, model_name):
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.6)
    lims = [min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))]
    ax.plot(lims, lims, linestyle="--", color="red")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{model_name} — Predicted vs Actual")
    st.pyplot(fig)


def plot_residuals(y_true, y_pred, model_name):
    residuals = y_true - y_pred
    fig, ax = plt.subplots()
    sns.histplot(residuals, kde=True, ax=ax)
    ax.set_title(f"{model_name} — Residuals Distribution")
    st.pyplot(fig)


def feature_importances_if_any(model, feature_names):
    # Works for RandomForest regressors
    try:
        importances = None
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "named_steps") and "model" in model.named_steps:
            m = model.named_steps["model"]
            if hasattr(m, "feature_importances_"):
                importances = m.feature_importances_
        if importances is not None:
            fi = pd.Series(importances, index=feature_names).sort_values(
                ascending=False
            )
            st.subheader("Feature importances")
            st.dataframe(
                fi.reset_index().rename(columns={"index": "feature", 0: "importance"})
            )
            fig, ax = plt.subplots(figsize=(6, max(3, len(fi) * 0.25)))
            fi.plot(kind="barh", ax=ax)
            ax.invert_yaxis()
            st.pyplot(fig)
    except Exception:
        pass


# --- Initialize session state ---
init_session_state()

# --- Title and description ---
st.markdown(
    """
    <div style="background-color:#f5f7fb;padding:18px;border-radius:8px">
      <h1 style="color:#0b3d91">🧠 Interactive Machine Learning Regression App</h1>
      <p style="font-size:14px;color:#333">
        Upload a CSV file (first row must be column headers). Explore data, select features/target, preprocess,
        split into train/test, and interactively train regression models. You can store trained models (in-session),
        compare metrics, and delete stored entries.
      </p>
      <ul style="color:#333">
        <li>CSV format: header row required.</li>
        <li>Only numeric columns are used automatically; non-numeric columns are ignored (you can convert externally).</li>
        <li>Stored models are kept in Streamlit session state for the current session.</li>
      </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# --- File upload ---
with st.expander("Upload CSV file", expanded=True):
    uploaded = st.file_uploader(
        "Upload a CSV file (first row should be headers)",
        type=["csv"],
        accept_multiple_files=False,
    )
    if uploaded is not None:
        df = load_csv(uploaded)
        if df is not None:
            st.session_state.df = df
            st.success(
                f"Loaded data with {df.shape[0]} rows and {df.shape[1]} columns."
            )
            with st.expander("Show raw CSV (first 5000 chars)"):
                try:
                    uploaded.seek(0)
                    raw = uploaded.read().decode("utf-8")
                    st.code(raw[:5000])
                except Exception:
                    pass

if st.session_state.df is None:
    st.info("Please upload a CSV file to continue.")
    st.stop()

df = st.session_state.df.copy()

# --- Data exploration container with tabs ---
with st.container():
    st.subheader("Data Exploration")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📋 Data Preview",
            "📊 Summary Stats",
            "📈 Distributions Plots",
            "Features vs Target",
            "Customize Plot",
        ]
    )

    with tab1:
        st.markdown("**Preview (first 200 rows)**")
        st.dataframe(df.head(200))
        st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

    with tab2:
        st.markdown("**Descriptive statistics**")
        try:
            # st.dataframe(df.describe(include="all").T)
            st.dataframe(describe(df))

        except Exception as e:
            st.write("Could not compute describe():", e)

        st.markdown("**Column types & null counts**")
        info_df = pd.DataFrame(
            {
                "dtype": df.dtypes.astype(str),
                "non-null": df.notnull().sum(),
                "null": df.isnull().sum(),
            }
        )
        st.dataframe(info_df)

    with tab3:
        st.markdown("**Visual exploration**")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.warning("No numeric columns in the uploaded dataset to plot.")
        else:
            col = st.selectbox(
                "Select numeric column to visualize", numeric_cols, index=0
            )
            st.write("Histogram and boxplot")
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))

            sns.histplot(
                df[col].dropna(),
                kde=True,
                ax=axes[0, 0],
                color="teal",
            )
            axes[0, 0].set_title(f"Histogram: {col}")

            sns.boxplot(
                df,
                x=col,
                ax=axes[0, 1],
                color="teal",
            )
            axes[0, 1].set_title(f"Box Plot: {col}")

            sns.violinplot(
                df[col].dropna(),
                orient="h",
                color="teal",
                ax=axes[1, 0],
            )
            axes[1, 0].set_title(f"Violin Plot: {col}")

            sns.ecdfplot(
                df[col].dropna(),
                ax=axes[1, 1],
                color="teal",
            )
            axes[1, 1].set_title(f"ECDF (Cumulative Distribution): {col}")
            fig.tight_layout()
            _, col, _ = st.columns([1, 3, 1])
            col.pyplot(fig)

    with tab4:
        output = st.selectbox("Target", df.columns, index=len(df.columns) - 1)

        # cols = st.columns(3)
        @st.cache_data
        def plot_features(df, output):
            inputs = df.columns.drop(output)
            for i in range(0, len(inputs), 3):
                cols = st.columns(3)
                for j, input in enumerate(inputs[i : i + 3]):
                    fig, ax = plt.subplots()
                    sns.scatterplot(df, x=input, y=output, ax=ax, edgecolor="k")
                    # ax.scatter(df[input].values, df[output].values)

                    cols[j].pyplot(fig)

        plot_features(df, output)
    with tab5:
        cols = st.columns(4)
        x_data = cols[0].selectbox("X axis", df.columns, index=0)
        y_data = cols[1].selectbox("Y axis", df.columns, index=len(df.columns) - 1)
        color = cols[2].color_picker("Marker Color", "#90D5FF")
        edgecolor = cols[3].color_picker("Marker Edge Color", "#000000")

        xlabel = cols[0].text_input("X Label", x_data)
        ylabel = cols[1].text_input("Y Label", y_data)
        cols[2].write("as")
        cols[3].write("asas")

        with st.container(border=False):
            cols = st.columns(4)
            xmin = cols[0].number_input("X axis min", value=None)
            xmax = cols[1].number_input("X axis max", value=None)
            ymin = cols[2].number_input("Y axis min", value=None)
            ymax = cols[3].number_input("Y axis max", value=None)
        if st.button("Plot"):
            fig = customize_plot(
                df,
                xdata=x_data,
                ydata=y_data,
                scatter_kwargs={"color": color, "edgecolor": edgecolor},
                ax_kwargs={
                    "xlabel": xlabel,
                    "ylabel": ylabel,
                    "xlim": (xmin, xmax),
                    "ylim": (ymin, ymax),
                },
            )
            _, col, _ = st.columns([1, 2, 1])
            col.pyplot(fig)


st.markdown("---")

# --- Preprocessing container ---
with st.container():
    st.subheader("Preprocessing & Train/Test Split")

    # Feature/target selection
    all_cols = df.columns.tolist()
    target_col = st.selectbox(
        "Select target column (y)", all_cols, index=len(all_cols) - 1
    )
    feature_cols = st.multiselect(
        "Select feature columns (X). If none selected, all numeric columns except target will be used",
        options=[c for c in all_cols if c != target_col],
        default=[
            c
            for c in df.select_dtypes(include=[np.number]).columns.tolist()
            if c != target_col
        ],
    )

    if not feature_cols:
        st.warning(
            "No feature columns selected. Select at least one feature column to proceed."
        )

    test_size = st.slider(
        "Test set size (fraction)", min_value=0.05, max_value=0.5, value=0.2, step=0.05
    )
    random_state = st.number_input("Random state (integer)", value=42, step=1)
    apply_split = st.button("Apply split / preview train/test")

    if apply_split:
        # Keep only numeric features automatically (imputer will handle missing)
        X = df[feature_cols]
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=int(random_state)
        )
        st.session_state.split = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "test_size": test_size,
            "random_state": int(random_state),
            "feature_cols": feature_cols,
            "target_col": target_col,
        }
        st.success(
            f"Split applied: train={X_train.shape[0]} rows, test={X_test.shape[0]} rows."
        )
        st.dataframe(pd.concat([X_train.head(), y_train.head()], axis=1))

# If no split in session_state, create a default one (not applied until user clicks)
if not st.session_state.split:
    # default split using numeric features (if any)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    default_features = [c for c in numeric_cols if c != df.columns[-1]]
    default_target = df.columns[-1]
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            df[default_features], df[default_target], test_size=0.2, random_state=42
        )
        st.session_state.split = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "test_size": 0.2,
            "random_state": 42,
            "feature_cols": default_features,
            "target_col": default_target,
        }
    except Exception:
        # can't auto-split (no numeric features), keep empty split - user must select
        pass

# If split still empty, stop
if not st.session_state.split:
    st.warning(
        "No valid train/test split available. Ensure you have numeric features and a numeric target."
    )
    st.stop()

split = st.session_state.split

st.markdown("---")

# --- Model training container ---
with st.container():
    st.subheader("Model Training")

    model_type = st.selectbox(
        "Choose regression algorithm",
        options=["Linear Regression", "Ridge", "Lasso", "Random Forest", "SVR"],
    )

    # dynamic hyperparameters
    st.markdown("**Hyperparameters**")
    # Defaults
    params = {}
    if model_type == "Linear Regression":
        st.markdown("No hyperparameters for basic Linear Regression.")
        use_scaler = True
    elif model_type == "Ridge":
        alpha = st.number_input(
            "alpha (regularization)", min_value=0.0, value=1.0, step=0.1
        )
        params["alpha"] = float(alpha)
        use_scaler = True
    elif model_type == "Lasso":
        alpha = st.number_input(
            "alpha (regularization)", min_value=0.0, value=0.1, step=0.1
        )
        params["alpha"] = float(alpha)
        use_scaler = True
    elif model_type == "Random Forest":
        n_estimators = st.number_input(
            "n_estimators", min_value=10, max_value=2000, value=100, step=10
        )
        max_depth = st.number_input(
            "max_depth (0 -> None)", min_value=0, max_value=100, value=0, step=1
        )
        min_samples_leaf = st.number_input(
            "min_samples_leaf", min_value=1, max_value=50, value=1, step=1
        )
        params.update(
            {
                "n_estimators": int(n_estimators),
                "max_depth": int(max_depth) if int(max_depth) > 0 else None,
                "min_samples_leaf": int(min_samples_leaf),
            }
        )
        use_scaler = False  # trees don't need scaling
    elif model_type == "SVR":
        C = st.number_input("C (regularization)", min_value=0.01, value=1.0, step=0.01)
        kernel = st.selectbox("kernel", options=["rbf", "linear", "poly"])
        params.update({"C": float(C), "kernel": kernel})
        use_scaler = True

    # Model name for storage key
    model_name_input = st.text_input(
        "Model name (used as key when storing). Default = selected algorithm",
        value=model_type,
    )

    # Buttons: Train, Store, Delete
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        if st.button("Train model (fit & show metrics)"):
            # build the model object
            try:
                X_train = split["X_train"][split["feature_cols"]]
                X_test = split["X_test"][split["feature_cols"]]
                y_train = split["y_train"]
                y_test = split["y_test"]

                if model_type == "Linear Regression":
                    model_obj = LinearRegression()
                elif model_type == "Ridge":
                    model_obj = Ridge(alpha=params["alpha"])
                elif model_type == "Lasso":
                    model_obj = Lasso(alpha=params["alpha"], max_iter=10000)
                elif model_type == "Random Forest":
                    model_obj = RandomForestRegressor(
                        n_estimators=params["n_estimators"],
                        max_depth=params["max_depth"],
                        min_samples_leaf=params["min_samples_leaf"],
                        random_state=int(split.get("random_state", 42)),
                    )
                elif model_type == "SVR":
                    model_obj = SVR(C=params["C"], kernel=params["kernel"])

                pipeline = build_pipeline(model_obj, use_scaler=use_scaler)
                pipeline.fit(X_train, y_train)
                preds = pipeline.predict(X_test)
                metrics = compute_metrics(y_test, preds)

                st.success("Model trained.")
                st.markdown("**Metrics on test set**")
                st.json(metrics)

                # Plots
                plot_predictions(y_test, preds, model_name_input)
                plot_residuals(y_test, preds, model_name_input)
                # Feature importances
                feature_importances_if_any(pipeline, split["feature_cols"])

                # keep last trained model in session state temporarily
                st.session_state.last_train = {
                    "name": model_name_input,
                    "type": model_type,
                    "pipeline": pipeline,
                    "metrics": metrics,
                    "params": params,
                    "timestamp": time.time(),
                }

            except Exception as e:
                st.error(f"Error during training: {e}")

    with col2:
        if st.button("Store model (save into session dictionary)"):
            if st.session_state.get("last_train") is None:
                st.warning(
                    "No model trained in this session to store. Train first, then store."
                )
            else:
                key = model_name_input
                # store a copy of the pipeline and metrics
                entry = {
                    "type": st.session_state.last_train.get("type"),
                    "pipeline": st.session_state.last_train.get("pipeline"),
                    "metrics": st.session_state.last_train.get("metrics"),
                    "params": st.session_state.last_train.get("params"),
                    "timestamp": st.session_state.last_train.get("timestamp"),
                }
                st.session_state.models[key] = entry
                st.success(f"Stored model under key: '{key}'")

    with col3:
        if st.button("Delete stored model"):
            key_to_delete = st.selectbox(
                "Select stored model key to delete",
                options=list(st.session_state.models.keys()) or ["(none)"],
            )
            # small confirm button
            if key_to_delete != "(none)":
                if st.button("Confirm delete"):
                    if key_to_delete in st.session_state.models:
                        del st.session_state.models[key_to_delete]
                        st.success(f"Deleted model '{key_to_delete}'.")
                    else:
                        st.warning("Key not found (it may have been deleted already).")

    with col4:
        if st.button("Download last trained model (pickle)"):
            if st.session_state.get("last_train") is None:
                st.warning("No model trained to download.")
            else:
                # create pickle bytes
                try:
                    pipeline = st.session_state.last_train["pipeline"]
                    b = pickle.dumps(pipeline)
                    st.download_button(
                        "Click to download .pkl",
                        data=b,
                        file_name=f"{st.session_state.last_train['name']}.pkl",
                        mime="application/octet-stream",
                    )
                except Exception as e:
                    st.error(f"Could not create pickle: {e}")

st.markdown("---")

# --- Stored models summary container ---
with st.container():
    st.subheader("Stored Models Summary")
    if not st.session_state.models:
        st.info(
            "No models stored yet. Train and store a model to see the summary here."
        )
    else:
        # Build summary DataFrame for display
        rows = []
        for k, v in st.session_state.models.items():
            row = {
                "model_key": k,
                "type": v.get("type"),
                "r2": v["metrics"].get("r2"),
                "mse": v["metrics"].get("mse"),
                "rmse": v["metrics"].get("rmse"),
                "mae": v["metrics"].get("mae"),
                "params": str(v.get("params")),
                "stored_at": pd.to_datetime(v.get("timestamp"), unit="s"),
            }
            rows.append(row)
        summary_df = pd.DataFrame(rows).sort_values(by="r2", ascending=False)
        st.dataframe(summary_df.reset_index(drop=True))

        st.markdown("**Compare models visually (R²)**")
        fig, ax = plt.subplots(figsize=(8, max(3, len(summary_df) * 0.4)))
        sns.barplot(x="r2", y="model_key", data=summary_df, ax=ax)
        ax.set_xlabel("R²")
        ax.set_ylabel("Model key")
        st.pyplot(fig)

st.markdown("---")
st.caption(
    "App created with Streamlit — Upload CSV with headers to begin. Stored models persist only for the current session."
)
