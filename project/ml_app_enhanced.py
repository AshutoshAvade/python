# ml_app.py
"""
Enhanced ML Playground — single-file Streamlit app with 6 pages.
Features:
- Upload CSV, EDA, advanced preprocessing (impute/encode/scale),
  supervised & unsupervised algorithms, CV, tuning, ensembles, SHAP explanations,
  clustering helpers (elbow, silhouette, dendrogram), and comparisons.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import time, warnings
warnings.filterwarnings("ignore")

# sklearn imports
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit, LeaveOneOut
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, silhouette_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    VotingClassifier, VotingRegressor, IsolationForest
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.manifold import TSNE

# Optional third-party models
HAS_XGBOOST = False
HAS_LIGHTGBM = False
HAS_CATBOOST = False
HAS_SHAP = False

try:
    import xgboost as xgb
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LIGHTGBM = True
except Exception:
    HAS_LIGHTGBM = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# Streamlit layout
st.set_page_config(page_title="ML Playground (Enhanced)", layout="wide")
st.title("ML Playground — Upload, Explore, Train & Compare")
st.markdown("Upload a CSV, explore the data, configure preprocessing, train supervised/unsupervised models, tune, visualize, and compare.")

# ---------------------------
# Utilities
# ---------------------------
@st.cache_data
def load_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding='latin1', engine='python')
    return df

def sample_df(df, n=500, random_state=42):
    if df.shape[0] <= n:
        return df.copy()
    return df.sample(n=n, random_state=random_state).reset_index(drop=True)

def infer_target(df):
    candidates = [c for c in df.columns if c.lower() in ("target", "label", "y", "class")]
    if candidates:
        return candidates[0]
    numeric = df.select_dtypes(include=[np.number]).nunique().sort_values()
    if not numeric.empty:
        for col, uniq in numeric.items():
            if 2 <= uniq <= max(100, df.shape[0] // 10):
                return col
    return df.columns[-1]

def detect_task(df, target_col):
    try:
        series = df[target_col]
    except Exception:
        return "unsupervised"
    if pd.api.types.is_numeric_dtype(series):
        unique = series.nunique()
        if unique > 20:
            return "regression"
        else:
            return "classification"
    else:
        return "classification"

def metric_summary(task, y_true, y_pred):
    if task == "classification":
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        return {"accuracy": round(acc, 4), "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}
    else:
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        ev = explained_variance_score(y_true, y_pred)
        return {"mse": round(mse, 4), "mae": round(mae, 4), "r2": round(r2, 4), "explained_variance": round(ev, 4)}

# Preprocessor builder
def build_preprocessor(
    df,
    target_col=None,
    numeric_impute_strategy="mean",
    categorical_impute_strategy="most_frequent",
    low_cardinality_threshold=10,
    onehot_handle_unknown="ignore",
    scale_numeric=True
):
    dfc = df.copy().dropna(axis=1, how='all')
    if target_col and target_col in dfc.columns:
        y = dfc[target_col].copy()
        X = dfc.drop(columns=[target_col])
    else:
        y = None
        X = dfc.copy()

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    cardinality = {c: X[c].nunique(dropna=True) for c in categorical_cols}
    low_card_cols = [c for c, card in cardinality.items() if card <= low_cardinality_threshold]
    high_card_cols = [c for c, card in cardinality.items() if card > low_cardinality_threshold]

    num_imputer = SimpleImputer(strategy=numeric_impute_strategy)
    cat_imputer = SimpleImputer(strategy=(categorical_impute_strategy if categorical_impute_strategy != "constant" else "constant"), fill_value="__missing__")

    transformers = []
    num_pipeline_steps = [("imputer", num_imputer)]
    if scale_numeric:
        num_pipeline_steps.append(("scaler", StandardScaler()))
    if numeric_cols:
        num_pipeline = Pipeline(num_pipeline_steps)
        transformers.append(("num", num_pipeline, numeric_cols))

    if low_card_cols:
        try:
            onehot = OneHotEncoder(handle_unknown=onehot_handle_unknown, sparse_output=False)
        except TypeError:
            onehot = OneHotEncoder(handle_unknown=onehot_handle_unknown, sparse=False)
        cat_low_pipeline = Pipeline([("imputer", cat_imputer), ("onehot", onehot)])
        transformers.append(("cat_low", cat_low_pipeline, low_card_cols))

    if high_card_cols:
        try:
            ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        except TypeError:
            ord_enc = OrdinalEncoder()
        cat_high_pipeline = Pipeline([("imputer", cat_imputer), ("ordinal", ord_enc)])
        transformers.append(("cat_high", cat_high_pipeline, high_card_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0)
    summary = {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "low_cardinality_cols": low_card_cols,
        "high_cardinality_cols": high_card_cols,
        "numeric_impute_strategy": numeric_impute_strategy,
        "categorical_impute_strategy": categorical_impute_strategy,
        "scale_numeric": scale_numeric,
        "onehot_handle_unknown": onehot_handle_unknown
    }
    return preprocessor, summary, X, y

# safe fit transform caching wrapper
@st.cache_data
def fit_preprocessor(_preprocessor, X):
    _preprocessor.fit(X)
    return _preprocessor

def transform_with_preprocessor(preprocessor, X):
    arr = preprocessor.transform(X)
    out_cols = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(trans, "named_steps"):
            last_step = list(trans.named_steps.items())[-1][1]
            if isinstance(last_step, OneHotEncoder):
                try:
                    cats = last_step.categories_
                    for col, categories in zip(cols, cats):
                        out_cols += [f"{col}__{str(cat)}" for cat in categories]
                except Exception:
                    out_cols += list(cols)
            else:
                out_cols += list(cols)
        else:
            out_cols += list(cols)
    try:
        return pd.DataFrame(arr, columns=out_cols, index=X.index)
    except Exception:
        return pd.DataFrame(arr, index=X.index)

# SHAP-safe wrapper
def show_shap_for_model(model, X_sample, max_display=10):
    if not HAS_SHAP:
        st.info("SHAP is not installed. To enable SHAP explanations install `shap` (pip install shap).")
        return
    # ensure model supports predict or predict_proba; otherwise skip
    if not hasattr(model, "predict"):
        st.info("SHAP explanation unavailable: selected model doesn't implement `predict()`.")
        return
    try:
        st.subheader("SHAP explanation (approx.)")
        # Choose TreeExplainer for tree models for speed, otherwise use KernelExplainer (slower)
        try:
            if hasattr(shap, "TreeExplainer") and (HAS_XGBOOST and isinstance(model, (xgb.XGBClassifier, xgb.XGBRegressor)) or
                                                   HAS_LIGHTGBM and isinstance(model, (lgb.LGBMClassifier, lgb.LGBMRegressor)) or
                                                   isinstance(model, (RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor))):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
            else:
                explainer = shap.Explainer(model.predict, X_sample)  # generic
                shap_values = explainer(X_sample)
            # summary plot
            plt.figure(figsize=(6,4))
            try:
                shap.summary_plot(shap_values, X_sample, show=False, max_display=max_display)
            except Exception:
                # fallback for some shap outputs
                shap.summary_plot(shap_values, X_sample, show=False)
            st.pyplot(plt.gcf())
            plt.clf()
        except Exception as e:
            st.warning(f"SHAP explanation encountered an error: {e}")
    except Exception as e:
        st.warning(f"SHAP failed: {e}")

# ---------------------------
# Sidebar & session init
# ---------------------------
st.sidebar.header("Data & Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
use_sample_for_eda = st.sidebar.checkbox("Use sampling for EDA (fast)", value=True)
sample_size = st.sidebar.number_input("Sample size for EDA/plots", min_value=100, max_value=5000, value=500, step=100)
random_state = int(st.sidebar.number_input("Random seed", value=42, step=1))
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "trained_models" not in st.session_state:
    st.session_state.trained_models = {}
if "preprocessor" not in st.session_state:
    st.session_state.preprocessor = None
if "prep_summary" not in st.session_state:
    st.session_state.prep_summary = {}

# load df
if uploaded_file is not None:
    df = load_csv(uploaded_file)
    st.session_state.raw_df = df
    st.sidebar.success(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} cols")
else:
    df = st.session_state.raw_df

PAGES = [
    "1. EDA",
    "2. Algorithms (Train)",
    "3. Predictions & Output",
    "4. Tuning & Ensembles",
    "5. Before/After Comparison",
    "6. Plotting & Visualizations"
]
page = st.sidebar.selectbox("Select Page", PAGES)

if df is None:
    st.info("Upload a CSV to get started.")
    st.stop()

# Target selection (global)
with st.expander("Target column selection (auto-detected)"):
    inferred = infer_target(df)
    target_options = ["None"] + list(df.columns)
    default_index = target_options.index(inferred) if inferred in target_options else 0
    target_col = st.selectbox("Target column", options=target_options, index=default_index)
    if target_col == "None":
        task_inferred = "unsupervised"
    else:
        task_inferred = detect_task(df, target_col)
    st.markdown(f"Detected task type: **{task_inferred}** — you can override on Page 2 if needed.")

# ---------------------------
# Page 1: EDA
# ---------------------------
if page == "1. EDA":
    st.header("Exploratory Data Analysis (EDA)")
    st.subheader("Dataset overview")
    st.write("Shape:", df.shape)
    if st.checkbox("Show raw data (first 200 rows)"):
        st.dataframe(df.head(200))

    eda_df = sample_df(df, n=sample_size, random_state=random_state) if use_sample_for_eda else df.copy()
    st.info(f"Using sampled data: {eda_df.shape[0]} rows for quick EDA.") if use_sample_for_eda else None

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", eda_df.shape[0])
    with col2:
        st.metric("Columns", eda_df.shape[1])
    with col3:
        st.metric("Missing cells", int(eda_df.isna().sum().sum()))

    st.subheader("Columns, types & missing %")
    missing_pct = (eda_df.isna().sum() / eda_df.shape[0] * 100).round(2)
    dtypes = pd.DataFrame({
        "column": eda_df.columns,
        "dtype": eda_df.dtypes.astype(str),
        "nunique": eda_df.nunique(),
        "missing_pct": missing_pct
    }).reset_index(drop=True)
    st.dataframe(dtypes)

    st.subheader("Categorical cardinality")
    cat_cols = eda_df.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        card = {c: eda_df[c].nunique(dropna=True) for c in cat_cols}
        card_df = pd.DataFrame.from_dict(card, orient='index', columns=['n_unique']).reset_index().rename(columns={'index':'column'})
        st.dataframe(card_df.sort_values('n_unique', ascending=True))
    else:
        st.info("No categorical columns detected.")

    st.subheader("Top correlations (numeric)")
    numeric = eda_df.select_dtypes(include=[np.number])
    if numeric.shape[1] >= 2:
        corr = numeric.corr().abs().unstack().sort_values(ascending=False)
        corr = corr[corr < 1].drop_duplicates().head(20)
        st.dataframe(corr.reset_index().rename(columns={"level_0":"var1","level_1":"var2",0:"abs_corr"}))
        st.plotly_chart(px.imshow(numeric.corr()), use_container_width=True)
    else:
        st.info("Not enough numeric columns for correlation matrix.")

    st.subheader("Distribution / counts")
    sel_col = st.selectbox("Choose column to visualize", options=list(eda_df.columns))
    if pd.api.types.is_numeric_dtype(eda_df[sel_col]):
        fig, ax = plt.subplots()
        sns.histplot(eda_df[sel_col].dropna(), kde=True, ax=ax)
        st.pyplot(fig)
    else:
        st.write(eda_df[sel_col].value_counts().head(20))
        st.bar_chart(eda_df[sel_col].value_counts().head(20))

# ---------------------------
# Page 2: Algorithms (Train)
# ---------------------------
elif page == "2. Algorithms (Train)":
    st.header("Algorithms — configure preprocessing & select models")
    # Task override
    task_choice = st.radio("Task type (auto-detected)", options=["Auto", "classification", "regression", "unsupervised"], index=0)
    if task_choice == "Auto":
        task = "unsupervised" if target_col == "None" else detect_task(df, target_col)
    else:
        task = task_choice
    st.markdown(f"Selected task: **{task}**")

    # Preprocessing UI
    st.subheader("Preprocessing configuration")
    col_a, col_b = st.columns(2)
    with col_a:
        numeric_impute_strategy = st.selectbox("Numeric imputer", options=["mean","median","most_frequent"], index=0)
        categorical_impute_strategy = st.selectbox("Categorical imputer", options=["most_frequent","constant"], index=0)
        if categorical_impute_strategy == "constant":
            fill_value = st.text_input("Fill value for categorical missing", value="__missing__")
        else:
            fill_value = None
    with col_b:
        low_cardinality_threshold = st.number_input("Low-cardinality threshold (one-hot if ≤)", min_value=2, max_value=500, value=10)
        onehot_unknown = st.selectbox("OneHotEncoder handle_unknown", options=["ignore","error"], index=0)
        scale_numeric = st.checkbox("Scale numeric features (StandardScaler)", value=True)

    if st.button("Build & preview preprocessor"):
        preproc, summary, X_raw, y_raw = build_preprocessor(df, target_col if target_col!="None" else None,
                                                           numeric_impute_strategy=numeric_impute_strategy,
                                                           categorical_impute_strategy=categorical_impute_strategy,
                                                           low_cardinality_threshold=int(low_cardinality_threshold),
                                                           onehot_handle_unknown=onehot_unknown,
                                                           scale_numeric=scale_numeric)
        fitted = fit_preprocessor(preproc, X_raw)
        st.session_state.preprocessor = fitted
        st.session_state.prep_summary = summary
        st.success("Preprocessor built & fitted.")
        st.json(summary)
        try:
            X_preview = transform_with_preprocessor(fitted, X_raw.head(50))
            st.dataframe(X_preview.head(50))
        except Exception as e:
            st.warning("Preview failed to build column names: " + str(e))

    if st.session_state.preprocessor is not None:
        st.info("Preprocessor in session")
        st.json(st.session_state.prep_summary)

    # CV & scoring
    st.subheader("Cross-validation settings")
    cv_type = st.selectbox("CV type", ["KFold","StratifiedKFold","ShuffleSplit","LeaveOneOut"], index=0)
    n_splits = st.number_input("Number of splits (folds)", min_value=2, max_value=50, value=5)
    shuffle_cv = st.checkbox("Shuffle during CV", value=True)
    scoring_metric = st.selectbox("Scoring metric (for CV)", ["accuracy","f1_macro","precision_macro","recall_macro","neg_mean_squared_error","r2"], index=0)
    use_cv_for_training = st.checkbox("Use cross-validation during training", value=True)

    # Algorithm options (supervised + unsupervised)
    st.subheader("Select algorithm(s) to train")
    supervised_class_algs = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=random_state),
        "SVM (RBF)": SVC(probability=True),
        "K-Neighbors": KNeighborsClassifier()
    }
    supervised_reg_algs = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=random_state),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, random_state=random_state),
        "SVR": SVR(),
        "K-Neighbors Regressor": KNeighborsRegressor()
    }
    # include optional external models if present
    if HAS_XGBOOST:
        supervised_class_algs["XGBoost (Classifier)"] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state)
        supervised_reg_algs["XGBoost (Regressor)"] = XGBRegressor(random_state=random_state)
    if HAS_LIGHTGBM:
        supervised_class_algs["LightGBM (Classifier)"] = LGBMClassifier(random_state=random_state)
        supervised_reg_algs["LightGBM (Regressor)"] = LGBMRegressor(random_state=random_state)
    if HAS_CATBOOST:
        supervised_class_algs["CatBoost (Classifier)"] = CatBoostClassifier(verbose=0, random_state=random_state)
        supervised_reg_algs["CatBoost (Regressor)"] = CatBoostRegressor(verbose=0, random_state=random_state)

    unsupervised_algs = {
        "KMeans": "KMeans",
        "DBSCAN": "DBSCAN",
        "AgglomerativeClustering": "AgglomerativeClustering",
        "GaussianMixture": "GaussianMixture",
        "IsolationForest": "IsolationForest",
        "PCA": "PCA",
        "t-SNE": "t-SNE"
    }

    if task == "classification":
        alg_options = {**supervised_class_algs}
    elif task == "regression":
        alg_options = {**supervised_reg_algs}
    else:
        alg_options = {**unsupervised_algs}

    chosen = st.multiselect("Pick algorithms", options=list(alg_options.keys()), default=list(alg_options.keys())[:2])
    test_size = st.slider("Test set fraction (supervised only)", 0.05, 0.5, 0.2)

    if st.button("Train selected models"):
        if not chosen:
            st.warning("Choose at least one algorithm.")
        else:
            # build preprocessor if not exists
            if st.session_state.preprocessor is None:
                st.info("Building default preprocessor...")
                preproc, summary, X_raw, y_raw = build_preprocessor(df, target_col if target_col!="None" else None)
                fitted = fit_preprocessor(preproc, X_raw)
                st.session_state.preprocessor = fitted
                st.session_state.prep_summary = summary
            else:
                fitted = st.session_state.preprocessor
                _, _, X_raw, y_raw = build_preprocessor(df, target_col if target_col!="None" else None)

            if task in ("classification","regression"):
                X_trans = transform_with_preprocessor(fitted, X_raw)
                y = y_raw
                if not pd.api.types.is_numeric_dtype(y):
                    le = LabelEncoder()
                    y_enc = le.fit_transform(y.astype(str))
                else:
                    y_enc = y.values
                X_train, X_test, y_train, y_test = train_test_split(X_trans, y_enc, test_size=test_size, random_state=random_state)
                st.write("Train samples:", X_train.shape[0], "Test samples:", X_test.shape[0])
            else:
                X_trans = transform_with_preprocessor(fitted, X_raw)
                st.write("Unsupervised transformed shape:", X_trans.shape)

            # CV splitter
            if use_cv_for_training and task in ("classification","regression"):
                if cv_type == "KFold":
                    cv_splitter = KFold(n_splits=int(n_splits), shuffle=shuffle_cv, random_state=random_state if shuffle_cv else None)
                elif cv_type == "StratifiedKFold":
                    cv_splitter = StratifiedKFold(n_splits=int(n_splits), shuffle=shuffle_cv, random_state=random_state if shuffle_cv else None) if task=="classification" else KFold(n_splits=int(n_splits), shuffle=shuffle_cv, random_state=random_state if shuffle_cv else None)
                elif cv_type == "ShuffleSplit":
                    cv_splitter = ShuffleSplit(n_splits=int(n_splits), test_size=0.2, random_state=random_state)
                else:
                    cv_splitter = LeaveOneOut()
            else:
                cv_splitter = None

            st.info("Running chosen algorithms...")
            progress = st.progress(0)
            for i, name in enumerate(chosen):
                try:
                    if task in ("classification","regression"):
                        model = alg_options[name]
                        with st.spinner(f"Training {name}..."):
                            model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        metrics = metric_summary(task, y_test, y_pred)
                        st.subheader(name)
                        st.write(metrics)
                        if task == "classification":
                            st.text(classification_report(y_test, y_pred, zero_division=0))
                            cm = confusion_matrix(y_test, y_pred)
                            fig, ax = plt.subplots()
                            sns.heatmap(cm, annot=True, fmt='d', ax=ax)
                            st.pyplot(fig)
                        else:
                            fig, ax = plt.subplots()
                            ax.scatter(y_test, y_pred, alpha=0.6)
                            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                            st.pyplot(fig)

                        if cv_splitter is not None:
                            try:
                                scores = cross_val_score(model, X_trans, y_enc, cv=cv_splitter, scoring=scoring_metric, n_jobs=-1)
                                st.write(f"CV {scoring_metric} mean: {np.mean(scores):.4f} (std {np.std(scores):.4f})")
                            except Exception as e:
                                st.warning("CV failed: " + str(e))

                        st.session_state.trained_models[name] = {"model": model, "metrics": metrics, "task": task}

                        # SHAP (only if supported)
                        if st.button(f"Show SHAP for {name}"):
                            try:
                                # sample rows for SHAP to speed
                                X_sample = X_trans.sample(min(200, X_trans.shape[0]), random_state=random_state) if isinstance(X_trans, pd.DataFrame) else pd.DataFrame(X_trans).sample(min(200, X_trans.shape[0]), random_state=random_state)
                                show_shap_for_model(model, X_sample)
                            except Exception as e:
                                st.warning("SHAP failed: " + str(e))

                    else:
                        # unsupervised flows
                        algo_key = alg_options[name]
                        st.subheader(name)
                        if algo_key == "KMeans":
                            k = st.number_input(f"k for {name}", 2, 20, 3, key=f"k_{i}")
                            km = KMeans(n_clusters=int(k), random_state=random_state)
                            labels = km.fit_predict(X_trans)
                            st.write("Cluster counts:", pd.Series(labels).value_counts().to_dict())
                            pca = PCA(n_components=2)
                            coords = pca.fit_transform(X_trans)
                            fig, ax = plt.subplots()
                            sc = ax.scatter(coords[:,0], coords[:,1], c=labels, cmap='tab10', alpha=0.8)
                            st.pyplot(fig)
                            if len(set(labels))>1 and not (len(set(labels))==2 and -1 in set(labels)):
                                try:
                                    sil = silhouette_score(X_trans, labels)
                                    st.write("Silhouette score:", round(sil,4))
                                except Exception:
                                    pass
                            st.session_state.trained_models[f"KMeans_k{k}"] = {"model": km, "metrics": {}, "task": "unsupervised"}
                        elif algo_key == "DBSCAN":
                            eps = st.number_input(f"eps for {name}", 0.1, 10.0, 0.5, step=0.1, key=f"eps_{i}")
                            min_samples = st.number_input(f"min_samples for {name}", 1, 50, 5, key=f"mins_{i}")
                            db = DBSCAN(eps=float(eps), min_samples=int(min_samples))
                            labels = db.fit_predict(X_trans)
                            st.write("Cluster counts (note -1 = noise):", pd.Series(labels).value_counts().to_dict())
                            try:
                                pca = PCA(n_components=2)
                                coords = pca.fit_transform(X_trans)
                                fig, ax = plt.subplots()
                                ax.scatter(coords[:,0], coords[:,1], c=labels, cmap='tab10', alpha=0.8)
                                st.pyplot(fig)
                            except Exception:
                                pass
                            # silhouette excluding noise
                            try:
                                lab_no_noise = labels[labels!=-1]
                                if len(set(lab_no_noise))>1:
                                    sil = silhouette_score(X_trans[labels!=-1], lab_no_noise)
                                    st.write("Silhouette (excluding noise):", round(sil,4))
                            except Exception:
                                pass
                            st.session_state.trained_models[f"DBSCAN_eps{eps}_min{min_samples}"] = {"model": db, "metrics": {}, "task": "unsupervised"}
                        elif algo_key == "AgglomerativeClustering":
                            n_clusters = st.number_input(f"n_clusters for {name}", 2, 20, 3, key=f"agg_{i}")
                            agg = AgglomerativeClustering(n_clusters=int(n_clusters))
                            labels = agg.fit_predict(X_trans)
                            st.write("Cluster counts:", pd.Series(labels).value_counts().to_dict())
                            # dendrogram sample
                            sample_n = min(300, X_trans.shape[0])
                            try:
                                Z = linkage(X_trans[:sample_n], method='ward')
                                fig, ax = plt.subplots(figsize=(8,4))
                                dendrogram(Z, truncate_mode='level', p=5)
                                st.pyplot(fig)
                            except Exception:
                                pass
                            st.session_state.trained_models[f"Agglomerative_k{n_clusters}"] = {"model": agg, "metrics": {}, "task": "unsupervised"}
                        elif algo_key == "GaussianMixture":
                            n_comp = st.number_input(f"components for {name}", 2, 20, 3, key=f"gmm_{i}")
                            gmm = GaussianMixture(n_components=int(n_comp), random_state=random_state)
                            labels = gmm.fit_predict(X_trans)
                            st.write("Cluster counts:", pd.Series(labels).value_counts().to_dict())
                            pca = PCA(n_components=2)
                            coords = pca.fit_transform(X_trans)
                            fig, ax = plt.subplots()
                            ax.scatter(coords[:,0], coords[:,1], c=labels, cmap='tab10', alpha=0.8)
                            st.pyplot(fig)
                            st.session_state.trained_models[f"GMM_comp{n_comp}"] = {"model": gmm, "metrics": {}, "task": "unsupervised"}
                        elif algo_key == "IsolationForest":
                            cont = st.slider(f"contamination for {name}", 0.01, 0.5, 0.05, step=0.01, key=f"iso_{i}")
                            iso = IsolationForest(contamination=float(cont), random_state=random_state)
                            labels = iso.fit_predict(X_trans)
                            st.write("Outlier counts:", pd.Series(labels).value_counts().to_dict())
                            try:
                                pca = PCA(n_components=2)
                                coords = pca.fit_transform(X_trans)
                                fig, ax = plt.subplots()
                                ax.scatter(coords[:,0], coords[:,1], c=labels, cmap='coolwarm', alpha=0.8)
                                st.pyplot(fig)
                            except Exception:
                                pass
                            st.session_state.trained_models[f"Isolation_{cont}"] = {"model": iso, "metrics": {}, "task": "unsupervised"}
                        elif algo_key == "PCA":
                            pca = PCA(n_components=2)
                            coords = pca.fit_transform(X_trans)
                            st.write("Explained variance ratio:", pca.explained_variance_ratio_)
                            fig, ax = plt.subplots()
                            ax.scatter(coords[:,0], coords[:,1], alpha=0.8)
                            st.pyplot(fig)
                            st.session_state.trained_models[f"PCA_2D"] = {"model": pca, "metrics": {}, "task": "unsupervised"}
                        elif algo_key == "t-SNE":
                            perp = st.number_input(f"perplexity for {name}", 5, 50, 30, key=f"tsne_{i}")
                            tsne = TSNE(n_components=2, perplexity=int(perp), random_state=random_state)
                            coords = tsne.fit_transform(X_trans)
                            fig, ax = plt.subplots()
                            ax.scatter(coords[:,0], coords[:,1], alpha=0.8)
                            st.pyplot(fig)
                            st.session_state.trained_models[f"tSNE_p{perp}"] = {"model": tsne, "metrics": {}, "task": "unsupervised"}
                        else:
                            st.warning(f"{name} not supported in unsupervised flow.")
                except Exception as ex:
                    st.error(f"Algorithm {name} failed: {ex}")
                progress.progress(int((i+1)/len(chosen)*100))
            st.success("Done. Trained/runs saved to session when applicable.")

# ---------------------------
# Page 3: Predictions & Output
# ---------------------------
elif page == "3. Predictions & Output":
    st.header("Predictions & Output")
    if not st.session_state.trained_models:
        st.info("No trained models in session. Train models on Page 2 first.")
        st.stop()

    model_names = list(st.session_state.trained_models.keys())
    chosen_name = st.selectbox("Choose a trained model", model_names)
    info = st.session_state.trained_models[chosen_name]
    st.subheader(chosen_name)
    st.write("Stored metadata/metrics (if any):")
    st.json(info.get("metrics", {}))

    if st.button("Re-evaluate model on fresh split (supervised only)"):
        if st.session_state.preprocessor is None:
            st.warning("No preprocessor in session.")
        else:
            _, _, X_raw, y_raw = build_preprocessor(df, target_col if target_col!="None" else None)
            X_trans = transform_with_preprocessor(st.session_state.preprocessor, X_raw)
            if y_raw is None:
                st.warning("This model appears unsupervised — can't re-evaluate with supervised split.")
            else:
                if not pd.api.types.is_numeric_dtype(y_raw):
                    le = LabelEncoder()
                    y_enc = le.fit_transform(y_raw.astype(str))
                else:
                    y_enc = y_raw.values
                X_train, X_test, y_train, y_test = train_test_split(X_trans, y_enc, test_size=0.2, random_state=random_state)
                model = info["model"]
                try:
                    y_pred = model.predict(X_test)
                    st.write(metric_summary(info.get("task"), y_test, y_pred))
                    if info.get("task")=="classification":
                        st.text(classification_report(y_test, y_pred, zero_division=0))
                        cm = confusion_matrix(y_test, y_pred)
                        fig, ax = plt.subplots()
                        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
                        st.pyplot(fig)
                    else:
                        fig, ax = plt.subplots()
                        ax.scatter(y_test, y_pred, alpha=0.6)
                        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                        st.pyplot(fig)
                except Exception as e:
                    st.error("Re-evaluation failed: " + str(e))

    st.subheader("Predict single sample (preprocessed)")
    if st.checkbox("Show input form to predict one sample"):
        if st.session_state.preprocessor is None:
            st.warning("Build preprocessor on Page 2 first.")
        else:
            _, _, X_raw, _ = build_preprocessor(df, target_col if target_col!="None" else None)
            raw_cols = X_raw.columns.tolist()
            input_vals = {}
            cols_per_row = 3
            cols_layout = [st.columns(cols_per_row) for _ in range((len(raw_cols)+cols_per_row-1)//cols_per_row)]
            for i, c in enumerate(raw_cols):
                r = i // cols_per_row; cc = i % cols_per_row
                default = ""
                try:
                    default = str(X_raw[c].mode().iloc[0])
                except Exception:
                    default = ""
                input_vals[c] = cols_layout[r][cc].text_input(c, value=default)
            if st.button("Predict sample"):
                row = pd.DataFrame([input_vals], columns=raw_cols)
                for c in row.columns:
                    try:
                        row[c] = pd.to_numeric(row[c])
                    except Exception:
                        row[c] = row[c].astype(str)
                Xs = transform_with_preprocessor(st.session_state.preprocessor, row)
                model = info["model"]
                try:
                    pred = model.predict(Xs)
                    st.success(f"Predicted: {pred[0]}")
                except Exception as e:
                    st.error("Prediction failed: " + str(e))

# ---------------------------
# Page 4: Tuning & Ensembles
# ---------------------------
elif page == "4. Tuning & Ensembles":
    st.header("Tuning & Ensembles")
    st.write("RandomizedSearchCV tuning for a chosen base model, plus voting ensembles.")

    cv_type = st.selectbox("CV type for tuning", ["KFold","StratifiedKFold","ShuffleSplit","LeaveOneOut"], index=0)
    n_splits = st.number_input("CV splits", 2, 50, 5)
    shuffle_cv = st.checkbox("Shuffle CV", value=True)
    scoring_metric = st.selectbox("Scoring for tuning", ["accuracy","f1_macro","precision_macro","recall_macro","neg_mean_squared_error","r2"], index=0)

    base_model_choice = st.selectbox("Base model to tune", ["RandomForestClassifier","GradientBoostingClassifier","RandomForestRegressor","GradientBoostingRegressor"])
    n_iter = st.number_input("RandomizedSearch n_iter", 5, 200, 20)
    run_tune = st.button("Run RandomizedSearchCV tuning")

    def default_param_dist(name):
        if "RandomForest" in name:
            return {"n_estimators":[50,100,200],"max_depth":[None,5,10,20],"min_samples_split":[2,5,10]}
        if "GradientBoosting" in name:
            return {"n_estimators":[50,100,200],"learning_rate":[0.01,0.05,0.1],"max_depth":[3,5,8]}
        return {}

    if run_tune:
        if st.session_state.preprocessor is None:
            st.warning("Build preprocessor on Page 2 first.")
        else:
            _, _, X_raw, y_raw = build_preprocessor(df, target_col if target_col!="None" else None)
            X_trans = transform_with_preprocessor(st.session_state.preprocessor, X_raw)
            if y_raw is None:
                st.warning("No target — tuning requires supervised data.")
            else:
                if not pd.api.types.is_numeric_dtype(y_raw):
                    le = LabelEncoder(); y_enc = le.fit_transform(y_raw.astype(str))
                else:
                    y_enc = y_raw.values
                if "Classifier" in base_model_choice:
                    model = RandomForestClassifier(random_state=random_state) if "RandomForest" in base_model_choice else GradientBoostingClassifier(random_state=random_state)
                    scoring = scoring_metric if scoring_metric in ["accuracy","f1_macro","precision_macro","recall_macro"] else "accuracy"
                else:
                    model = RandomForestRegressor(random_state=random_state) if "RandomForest" in base_model_choice else GradientBoostingRegressor(random_state=random_state)
                    scoring = scoring_metric if scoring_metric in ["neg_mean_squared_error","r2"] else "neg_mean_squared_error"

                if cv_type=="KFold":
                    cv_splitter = KFold(n_splits=int(n_splits), shuffle=shuffle_cv, random_state=random_state if shuffle_cv else None)
                elif cv_type=="StratifiedKFold":
                    cv_splitter = StratifiedKFold(n_splits=int(n_splits), shuffle=shuffle_cv, random_state=random_state if shuffle_cv else None)
                elif cv_type=="ShuffleSplit":
                    cv_splitter = ShuffleSplit(n_splits=int(n_splits), test_size=0.2, random_state=random_state)
                else:
                    cv_splitter = LeaveOneOut()

                param_dist = default_param_dist(base_model_choice)
                if not param_dist:
                    param_dist = {"n_estimators":[50,100], "max_depth":[None,5,10]}

                rs = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=int(n_iter), cv=cv_splitter, scoring=scoring, random_state=random_state, n_jobs=-1)
                st.info("Running RandomizedSearchCV...")
                rs.fit(X_trans, y_enc)
                best = rs.best_estimator_
                st.success("Tuning complete.")
                st.write("Best params:", rs.best_params_)
                X_train, X_test, y_train, y_test = train_test_split(X_trans, y_enc, test_size=0.2, random_state=random_state)
                y_pred = best.predict(X_test)
                metrics = metric_summary("classification" if "Classifier" in base_model_choice else "regression", y_test, y_pred)
                st.write("Performance on held-out set after tuning:", metrics)
                tuned_name = base_model_choice + "_tuned"
                st.session_state.trained_models[tuned_name] = {"model": best, "metrics": metrics, "task": "classification" if "Classifier" in base_model_choice else "regression"}
                st.success(f"Tuned saved as {tuned_name}")

    st.markdown("---")
    st.subheader("Voting Ensemble")
    ensemble_members = st.multiselect("Pick trained models", options=list(st.session_state.trained_models.keys()))
    ensemble_name = st.text_input("Ensemble name", value="ensemble_voting")
    ensemble_type = st.selectbox("Ensemble type", ["classification","regression"])
    if st.button("Create & evaluate ensemble"):
        if len(ensemble_members)<2:
            st.warning("Choose at least 2 models.")
        else:
            estimators = [(m, st.session_state.trained_models[m]["model"]) for m in ensemble_members]
            if ensemble_type=="classification":
                ens = VotingClassifier(estimators=estimators, voting="soft")
            else:
                ens = VotingRegressor(estimators=estimators)
            if st.session_state.preprocessor is None:
                st.warning("Build preprocessor first.")
            else:
                _, _, X_raw, y_raw = build_preprocessor(df, target_col if target_col!="None" else None)
                X_trans = transform_with_preprocessor(st.session_state.preprocessor, X_raw)
                if y_raw is None:
                    st.warning("No target available for ensemble evaluation.")
                else:
                    if not pd.api.types.is_numeric_dtype(y_raw):
                        le = LabelEncoder(); y_enc = le.fit_transform(y_raw.astype(str))
                    else:
                        y_enc = y_raw.values
                    X_train, X_test, y_train, y_test = train_test_split(X_trans, y_enc, test_size=0.2, random_state=random_state)
                    ens.fit(X_train, y_train)
                    y_pred = ens.predict(X_test)
                    metrics = metric_summary(ensemble_type, y_test, y_pred)
                    st.write("Ensemble performance:", metrics)
                    st.session_state.trained_models[ensemble_name] = {"model": ens, "metrics": metrics, "task": ensemble_type}
                    st.success(f"Ensemble {ensemble_name} saved.")

# ---------------------------
# Page 5: Comparison
# ---------------------------
elif page == "5. Before/After Comparison":
    st.header("Before vs After Comparison")
    comparisons = []
    for name, info in st.session_state.trained_models.items():
        row = {"model": name, "task": info.get("task")}
        row.update(info.get("metrics", {}))
        if "cv_mean" in info:
            row["cv_mean"] = info["cv_mean"]; row["cv_std"]=info.get("cv_std","")
        comparisons.append(row)
    if not comparisons:
        st.info("No trained models saved yet.")
        st.stop()
    df_comp = pd.DataFrame(comparisons).fillna("-")
    st.dataframe(df_comp)
    if st.button("Download comparison CSV"):
        st.download_button("Download CSV", data=df_comp.to_csv(index=False).encode('utf-8'), file_name="comparison.csv", mime="text/csv")

# ---------------------------
# Page 6: Visualizations
# ---------------------------
elif page == "6. Plotting & Visualizations":
    st.header("Plotting & Visualizations")
    st.write("Quick plots, PCA, and clustering helpers (elbow/silhouette/dendrogram).")

    viz_type = st.selectbox("Plot type", ["Scatter","Histogram","Boxplot","Correlation Heatmap","PCA 2D","KMeans Elbow/Silhouette/Dendrogram"])
    cols = df.columns.tolist()
    if viz_type=="Scatter":
        x = st.selectbox("X-axis", cols, index=0)
        y = st.selectbox("Y-axis", cols, index=1)
        color = st.selectbox("Color (optional)", ["None"]+cols)
        if st.button("Render scatter"):
            fig = px.scatter(df, x=x, y=y, color=None if color=="None" else color, title=f"{x} vs {y}")
            st.plotly_chart(fig, use_container_width=True)
    elif viz_type=="Histogram":
        col = st.selectbox("Column", cols)
        bins = st.slider("Bins", 5, 200, 30)
        if st.button("Render histogram"):
            fig = px.histogram(df, x=col, nbins=bins, title=f"Histogram of {col}")
            st.plotly_chart(fig, use_container_width=True)
    elif viz_type=="Boxplot":
        col = st.selectbox("Column", cols)
        by = st.selectbox("Group by (optional)", ["None"]+cols)
        if st.button("Render boxplot"):
            fig = px.box(df, y=col, color=None if by=="None" else by, title=f"Boxplot of {col}")
            st.plotly_chart(fig, use_container_width=True)
    elif viz_type=="Correlation Heatmap":
        numeric = df.select_dtypes(include=[np.number])
        if numeric.shape[1] < 2:
            st.info("Not enough numeric columns.")
        else:
            if st.button("Show correlation heatmap"):
                st.plotly_chart(px.imshow(numeric.corr(), text_auto=True, title="Correlation matrix"), use_container_width=True)
    elif viz_type=="PCA 2D":
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        use_cols = st.multiselect("Numeric columns to use (max 10)", numeric_cols, default=numeric_cols[:6] if numeric_cols else [])
        if st.button("Run PCA 2D"):
            if len(use_cols) < 2:
                st.info("Choose at least 2 numeric columns.")
            else:
                d = df[use_cols].dropna()
                pca = PCA(n_components=2)
                coords = pca.fit_transform(StandardScaler().fit_transform(d))
                plot_df = pd.DataFrame(coords, columns=["PC1","PC2"])
                st.plotly_chart(px.scatter(plot_df, x="PC1", y="PC2", title="PCA 2D"), use_container_width=True)
    else:
        st.write("Clustering helpers")
        # build X_vis
        if st.session_state.preprocessor is not None:
            _, _, X_raw_vis, _ = build_preprocessor(df, target_col if target_col!="None" else None)
            X_vis = transform_with_preprocessor(st.session_state.preprocessor, X_raw_vis)
            try:
                X_vis_arr = X_vis.values if isinstance(X_vis, pd.DataFrame) else X_vis
            except Exception:
                X_vis_arr = np.asarray(X_vis)
        else:
            numeric = df.select_dtypes(include=[np.number]).dropna()
            if numeric.shape[1]==0:
                st.warning("No numeric columns.")
                st.stop()
            X_vis_arr = StandardScaler().fit_transform(numeric)

        max_k = st.slider("Max K for elbow", 2, 20, 10)
        if st.button("Run elbow + silhouette"):
            Ks = list(range(2, max_k+1))
            inertias = []; silhouettes = []
            for k in Ks:
                try:
                    km = KMeans(n_clusters=k, random_state=random_state)
                    labs = km.fit_predict(X_vis_arr)
                    inertias.append(km.inertia_)
                    try:
                        silhouettes.append(silhouette_score(X_vis_arr, labs))
                    except Exception:
                        silhouettes.append(np.nan)
                except Exception:
                    inertias.append(np.nan); silhouettes.append(np.nan)
            fig, ax = plt.subplots(1,2, figsize=(12,4))
            ax[0].plot(Ks, inertias, marker='o'); ax[0].set_title("Elbow (inertia)")
            ax[1].plot(Ks, silhouettes, marker='o'); ax[1].set_title("Silhouette")
            st.pyplot(fig)

        if st.button("Show dendrogram (sample up to 500 rows)"):
            sample_n = min(500, X_vis_arr.shape[0])
            try:
                Z = linkage(X_vis_arr[:sample_n], method='ward')
                fig, ax = plt.subplots(figsize=(12,4))
                dendrogram(Z, truncate_mode='level', p=5)
                st.pyplot(fig)
            except Exception as e:
                st.error("Dendrogram failed: " + str(e))

st.caption("If you need added features (explainability reports, SHAP on large datasets, XGBoost/LightGBM hyperparam grids), tell me and I'll extend this app.")
