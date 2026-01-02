# ml_app.py
"""
Auto ML Streamlit app with EDA, Algorithms, Optimization, Comparison,
AND a "Suggest best algorithm" feature that benchmarks a handful of models
on a sample of your dataset and returns ranked recommendations.

This corrected version adds robust upload handling (avoids NoneType.shape error),
safe sample loading, defensive preprocessing, and stores preprocessors/models
in session state so pages that depend on them don't break.
"""
import io
import time
import warnings
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, confusion_matrix, r2_score, mean_squared_error, mean_absolute_error)

# Supervised algorithms
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor, GradientBoostingClassifier, GradientBoostingRegressor

# Unsupervised
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA

# optional xgboost
try:
    from xgboost import XGBClassifier, XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

warnings.filterwarnings("ignore")
st.set_page_config(layout="wide", page_title="Auto ML App + Suggestor")

# -------------------------
# Helpers
# -------------------------
def infer_task(y):
    """Infer if it's classification or regression based on y dtype and unique counts."""
    if y is None:
        return None
    try:
        # object/category => classification
        if getattr(y, "dtype", None) is not None and (y.dtype == object or y.dtype.name == 'category'):
            return "classification"
        # numeric heuristics
        unique = len(np.unique(y[~pd.isna(y)]))
        if unique <= 20 and (np.all(np.equal(np.mod(y.dropna(), 1), 0)) or unique < 10):
            return "classification"
        return "regression"
    except Exception:
        return "classification"

def auto_target_detect(df):
    for cand in ['target', 'label', 'y', 'class']:
        if cand in df.columns:
            return cand
    # fallback: last column if df not empty
    if df.shape[1] >= 1:
        return df.columns[-1]
    return None

def compute_classification_metrics(y_true, y_pred, y_prob=None):
    out = {}
    try:
        out['accuracy'] = float(accuracy_score(y_true, y_pred))
        out['precision'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
        out['recall'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
        out['f1'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    except Exception:
        out['accuracy'] = out['precision'] = out['recall'] = out['f1'] = None
    if y_prob is not None:
        try:
            out['roc_auc'] = float(roc_auc_score(pd.get_dummies(y_true).values, y_prob, average='macro', multi_class='ovr'))
        except Exception:
            out['roc_auc'] = None
    else:
        out['roc_auc'] = None
    return out

def compute_regression_metrics(y_true, y_pred):
    out = {}
    try:
        out['r2'] = float(r2_score(y_true, y_pred))
        out['rmse'] = float(mean_squared_error(y_true, y_pred, squared=False))
        out['mae'] = float(mean_absolute_error(y_true, y_pred))
    except Exception:
        out['r2'] = out['rmse'] = out['mae'] = None
    return out

def basic_preprocess(df, target_col=None, scaler=None, encode_categoricals=True):
    """Return X (features), y (target or None), preprocessor (ColumnTransformer), numeric_cols, cat_cols."""
    df = df.copy()
    if target_col and target_col in df.columns:
        y = df[target_col].copy()
        X = df.drop(columns=[target_col]).copy()
    else:
        X = df.copy()
        y = None

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    numeric_steps = [('imputer', SimpleImputer(strategy='mean'))]
    if scaler == 'standard':
        numeric_steps.append(('scaler', StandardScaler()))
    elif scaler == 'minmax':
        numeric_steps.append(('scaler', MinMaxScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)
    cat_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent'))])

    if encode_categoricals and len(cat_cols) > 0:
        from sklearn.preprocessing import OneHotEncoder
        # sparse=False for compatibility with DataFrame conversion; if large data, consider sparse=True
        cat_pipeline.steps.append(('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False)))

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_pipeline, numeric_cols),
        ('cat', cat_pipeline, cat_cols),
    ], remainder='drop')

    return X, y, preprocessor, numeric_cols, cat_cols

# Candidate models (for training & suggestor)
SUPERVISED_CLASSIFIERS = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "SVM (RBF)": SVC(probability=True),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier()
}
SUPERVISED_REGRESSORS = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "KNN Regressor": KNeighborsRegressor(),
    "SVR": SVR(),
    "Decision Tree Regressor": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(n_jobs=-1),
    "Gradient Boosting Regressor": GradientBoostingRegressor()
}
if _HAS_XGB:
    SUPERVISED_CLASSIFIERS["XGBoost Classifier"] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0)
    SUPERVISED_REGRESSORS["XGBoost Regressor"] = XGBRegressor()

# Light-weight benchmarking function used by suggestion feature
def benchmark_models_for_suggestion(df, target_col, task, preprocessor, sample_size=1000, cv=3, random_state=42, max_models=6):
    # defensive checks
    if df is None or target_col not in df.columns:
        return [], pd.DataFrame()

    # Sample dataset (stratify for classification if possible)
    if sample_size is not None and df.shape[0] > sample_size:
        try:
            if task == 'classification':
                # try stratified group sampling preserving classes roughly
                df_sample = df.groupby(target_col, group_keys=False).apply(
                    lambda x: x.sample(n=max(1, int(sample_size * len(x) / len(df))), random_state=random_state))
                df_sample = df_sample.sample(n=min(sample_size, len(df_sample)), random_state=random_state)
            else:
                df_sample = df.sample(n=sample_size, random_state=random_state)
        except Exception:
            df_sample = df.sample(n=sample_size, random_state=random_state)
    else:
        df_sample = df.copy()

    # ensure no all-NaN target rows
    df_sample = df_sample.dropna(subset=[target_col])
    if df_sample.shape[0] < 2:
        return [], df_sample

    X_raw = df_sample.drop(columns=[target_col])
    y_raw = df_sample[target_col]

    # preprocessor fit_transform
    try:
        X = preprocessor.fit_transform(X_raw)
    except Exception:
        # if preprocessor fails (rare), do naive numeric-only
        X = X_raw.select_dtypes(include=[np.number]).fillna(0).values

    results = []
    if task == 'classification':
        candidates = list(SUPERVISED_CLASSIFIERS.items())[:max_models]
        scoring = 'accuracy'
    else:
        candidates = list(SUPERVISED_REGRESSORS.items())[:max_models]
        scoring = 'r2'

    for name, model in candidates:
        pipe = Pipeline([('clf', model)])
        try:
            # cross_val_score: smaller cv for speed; n_jobs=-1 to utilize cores
            scores = cross_val_score(pipe, X, y_raw, cv=cv, scoring=scoring, n_jobs=-1)
            mean_score = float(np.mean(scores))
            std_score = float(np.std(scores))
            results.append({'model': name, 'mean_score': mean_score, 'std_score': std_score, 'raw_scores': [float(s) for s in scores]})
        except Exception as e:
            # capture error for debugging but continue
            results.append({'model': name, 'mean_score': None, 'std_score': None, 'error': str(e)})

    # Rank results (descending)
    results_sorted = sorted([r for r in results if r.get('mean_score') is not None], key=lambda x: x['mean_score'], reverse=True)
    return results_sorted, df_sample

def suggest_based_on_meta(df, target_col=None):
    # meta heuristics
    if df is None:
        return {}, []
    n_rows, n_cols = df.shape
    numeric_count = df.select_dtypes(include=[np.number]).shape[1]
    cat_count = df.select_dtypes(include=['object', 'category', 'bool']).shape[1]
    meta = {
        'n_rows': int(n_rows),
        'n_cols': int(n_cols),
        'numeric_count': int(numeric_count),
        'cat_count': int(cat_count),
        'numeric_ratio': float(numeric_count / max(1, n_cols)),
        'sparse_features': bool(cat_count > 0 and n_rows / max(1, cat_count) > 5)
    }
    recommendations = []
    if n_rows < 500:
        recommendations.append("Simple models (Logistic/Linear, Decision Tree, KNN) because dataset is small.")
    else:
        recommendations.append("Ensemble models (Random Forest, Gradient Boosting, XGBoost) may perform better on larger datasets.")
    if cat_count > 0 and numeric_count == 0:
        recommendations.append("Use models that handle categorical data after encoding (tree-based models are robust).")
    if meta['numeric_ratio'] > 0.8:
        recommendations.append("Numeric-heavy: consider scaling and algorithms like SVM / KNN / linear models.")
    if n_cols > 100:
        recommendations.append("High-dimensional data: prefer regularized linear models (Ridge/Lasso) or tree ensembles.")
    return meta, recommendations

# -------------------------
# Streamlit UI
# -------------------------
st.title("Auto ML: Upload → EDA → Models → Optimize → Suggestor")
st.write("Upload CSV or use sample dataset. New: 'Suggest best algorithm' runs a fast benchmark and returns ranked suggestions.")

# Sidebar inputs
with st.sidebar:
    st.header("Data")
    uploaded_file = st.file_uploader("Upload CSV or TXT", type=["csv", "txt", "xlsx", "xls"])
    sample_data = st.checkbox("Use sample dataset (Iris/Diabetes)", value=False)
    sample_type = None
    if sample_data:
        sample_type = st.selectbox("Sample dataset", ["iris (classification)", "diabetes (regression)"])
    st.markdown("---")
    page = st.radio("Choose page", ["1 — EDA", "2 — Algorithms & Suggestor", "3 — Evaluation & Optimization", "4 — Comparison"])

@st.cache_data
def load_data_from_buffer(buffer, sample_type=None):
    # If buffer is None but sample_type provided, load sklearn datasets
    if buffer is None and sample_type:
        try:
            if sample_type.startswith("iris"):
                from sklearn.datasets import load_iris
                d = load_iris(as_frame=True)
                df = d.frame.copy()
                df['target'] = d.target
                return df
            elif sample_type.startswith("diabetes"):
                from sklearn.datasets import load_diabetes
                d = load_diabetes(as_frame=True)
                df = d.frame.copy()
                df['target'] = d.target
                return df
        except Exception:
            return None

    if buffer is None:
        return None

    # buffer may be BytesIO or UploadedFile
    try:
        # if uploaded xlsx
        name = getattr(buffer, "name", "") or ""
        if name.lower().endswith((".xls", ".xlsx")):
            return pd.read_excel(buffer)
        else:
            return pd.read_csv(buffer)
    except Exception:
        try:
            return pd.read_table(buffer)
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            return None

# Load dataset (safe)
if sample_data and not uploaded_file:
    df = load_data_from_buffer(None, sample_type=sample_type)
elif uploaded_file:
    df = load_data_from_buffer(uploaded_file, sample_type=None)
else:
    df = None

# If no df, only page 1 allowed (or show message)
if df is None:
    st.info("No data loaded. Upload a CSV/TXT/XLSX or select a sample dataset in the sidebar.")
    # If user explicitly picked EDA page, allow limited display (no df)
    if page != "1 — EDA":
        st.stop()
else:
    # ensure df is a DataFrame
    if not isinstance(df, pd.DataFrame):
        st.error("Loaded object is not a DataFrame.")
        st.stop()
    # show success
    st.success(f"Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# session store (keep df + other objects)
if "store" not in st.session_state:
    st.session_state.store = {"df": df, "preprocessor": None, "trained_models": {}, "results": {}}
else:
    st.session_state.store["df"] = df

# ---------- Page 1: EDA ----------
if page == "1 — EDA":
    st.header("Exploratory Data Analysis")
    st.subheader("Data preview")
    if st.checkbox("Show raw data (first 200 rows)", False):
        st.dataframe(df.head(200))

    st.write("Shape:", df.shape)
    st.write("Columns & types")
    st.table(pd.DataFrame({"column": df.columns, "dtype": df.dtypes}))

    st.write("Missing values (top 20)")
    mv = df.isnull().sum().sort_values(ascending=False)
    st.bar_chart(mv.head(20))

    st.subheader("Descriptive statistics")
    try:
        st.write(df.describe(include='all').T)
    except Exception:
        st.write("Could not compute describe() for this dataset.")

# ---------- Page 2: Algorithms & Suggestor ----------
elif page == "2 — Algorithms & Suggestor":
    st.header("Algorithms — Prepare data & Suggest best algorithm")
    st.write("Select target column (for supervised). Preprocessing options. Use the Suggestor to get quick ranked recommendations.")

    # safe selectbox default: pick 'target' if present, else last column
    suggested_target = 'target' if 'target' in df.columns else (df.columns[-1] if len(df.columns) > 0 else None)
    options = [None] + list(df.columns)
    try:
        index_default = options.index(suggested_target) if suggested_target in options else len(options)-1
    except Exception:
        index_default = len(options)-1

    target_col = st.selectbox("Select target column (auto-detected)", options=options, index=index_default)
    if target_col is None:
        detected_task = None
        st.info("No target selected — unsupervised workflows will be available.")
    else:
        y = df[target_col]
        detected_task = infer_task(y)
        st.info(f"Detected task for `{target_col}`: {detected_task}")

    scaler_choice = st.selectbox("Scaler for numeric columns", ["None", "standard", "minmax"])
    encode_cat = st.checkbox("One-hot encode categorical features", value=True)

    # Build preprocessor and store it
    try:
        X, y, preprocessor, num_cols, cat_cols = basic_preprocess(df, target_col, scaler=(scaler_choice if scaler_choice != "None" else None), encode_categoricals=encode_cat)
        st.session_state.store["preprocessor"] = preprocessor
    except Exception as e:
        st.error(f"Failed to build preprocessor: {e}")
        X, y, preprocessor, num_cols, cat_cols = df.copy(), None, None, [], []

    st.subheader("Preview processed features (first 10 rows)")
    if preprocessor is not None:
        try:
            X_proc = preprocessor.fit_transform(X)
            # convert to DataFrame when possible
            if hasattr(X_proc, "shape"):
                st.dataframe(pd.DataFrame(X_proc).head(10))
        except Exception as e:
            st.warning(f"Preprocessing preview failed: {e}")
    else:
        st.write("Preprocessor not available — try adjusting preprocessing options.")

    st.markdown("---")
    st.subheader("Quick suggestion: recommend best algorithms")
    st.write("The suggestor runs a **light** benchmark (sample ≤ 1000 rows, cv=3) on a set of candidate models and ranks them.")
    sample_size = st.slider("Suggestion: sample size for benchmarking (smaller = faster)", min_value=100, max_value=2000, value=800, step=100)
    cv = st.slider("CV folds for benchmark", min_value=2, max_value=5, value=3)
    max_models = st.slider("Number of candidate models to benchmark", min_value=3, max_value=8, value=6)

    if st.button("Suggest best algorithm"):
        if target_col is None:
            st.info("Unsupervised dataset. Offering heuristic suggestions.")
            meta, recs = suggest_based_on_meta(df)
            st.write("Dataset meta:", meta)
            st.write("Heuristic recommendations:")
            for r in recs:
                st.write("-", r)
            st.write("Common unsupervised choices: PCA for dimensionality reduction, KMeans for clustering (if numeric), DBSCAN for density-based clusters.")
        else:
            if preprocessor is None:
                st.error("Preprocessor not available. Adjust preprocessing options first.")
            else:
                with st.spinner("Running lightweight benchmarks..."):
                    try:
                        results_sorted, df_sample = benchmark_models_for_suggestion(df, target_col, detected_task, preprocessor, sample_size=sample_size, cv=cv, max_models=max_models)
                        if len(results_sorted) == 0:
                            st.warning("No successful benchmark results (models failed). See errors below or reduce sample size.")
                        else:
                            st.success("Benchmark complete — top recommendations:")
                            top_n = min(3, len(results_sorted))
                            for i, r in enumerate(results_sorted[:top_n], start=1):
                                st.markdown(f"**#{i} — {r['model']}** — mean score: {r['mean_score']:.4f} ± {r['std_score']:.4f}")
                                if detected_task == 'classification':
                                    st.write("Note: scores are CV accuracy (higher better).")
                                else:
                                    st.write("Note: scores are CV R² (higher better).")
                            st.markdown("### Full benchmark results")
                            br = pd.DataFrame(results_sorted)
                            st.dataframe(br)

                            meta, recs = suggest_based_on_meta(df)
                            st.markdown("### Heuristic rationale")
                            st.write("Dataset meta:", meta)
                            st.write("Heuristic notes:")
                            for r in recs:
                                st.write("-", r)

                            # Save suggestion to session
                            st.session_state.store['suggestions'] = results_sorted
                            st.session_state.store['suggest_sample'] = df_sample
                    except Exception as e:
                        st.error(f"Suggestor failed: {e}")

    st.markdown("---")
    st.subheader("Train selected models (quick)")
    if detected_task == "classification":
        available = list(SUPERVISED_CLASSIFIERS.keys())
    elif detected_task == "regression":
        available = list(SUPERVISED_REGRESSORS.keys())
    else:
        available = []

    chosen = st.multiselect("Pick model(s) to train now (quick)", options=available, default=available[:2] if len(available) else [])
    test_size = st.slider("Test fraction", 0.05, 0.5, 0.2)
    rs = st.number_input("Random state", value=42)

    if st.button("Train selected models (quick)"):
        if target_col is None:
            st.warning("No target selected for supervised training.")
        else:
            if preprocessor is None:
                st.error("Preprocessor missing — cannot train. Go back and set preprocessing.")
            else:
                try:
                    X_full = preprocessor.fit_transform(X)
                except Exception:
                    X_full = X.select_dtypes(include=[np.number]).fillna(0).values
                try:
                    X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=test_size, random_state=rs, stratify=(y if detected_task=='classification' else None))
                except Exception:
                    X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=test_size, random_state=rs)

                models_trained = {}
                results = {}
                for name in chosen:
                    try:
                        if detected_task == "classification":
                            clf = SUPERVISED_CLASSIFIERS[name]
                        else:
                            clf = SUPERVISED_REGRESSORS[name]
                        clf.fit(X_train, y_train)
                        y_pred = clf.predict(X_test)
                        if detected_task == "classification":
                            y_prob = clf.predict_proba(X_test) if hasattr(clf, "predict_proba") else None
                            metrics = compute_classification_metrics(y_test, y_pred, y_prob=y_prob)
                        else:
                            metrics = compute_regression_metrics(y_test, y_pred)
                        models_trained[name] = clf
                        results[name] = metrics
                    except Exception as e:
                        st.error(f"Failed to train {name}: {e}")
                st.session_state.store["trained_models"] = models_trained
                st.session_state.store["results"] = results
                st.success("Training finished and stored.")
                for name, metrics in results.items():
                    st.write(f"**{name}**")
                    st.json(metrics)

# Page 3: Evaluation & Optimization
elif page == "3 — Evaluation & Optimization":
    st.header("Evaluation & Hyperparameter Search")
    preprocessor = st.session_state.store.get("preprocessor", None)
    if preprocessor is None:
        st.warning("Set preprocessing in Page 2 first.")
        st.stop()

    target_col = st.selectbox("Select target column for optimization", options=[None] + list(df.columns), index=(len(df.columns) if 'target' not in df.columns else list(df.columns).index('target')+1))
    if target_col is None:
        st.info("Optimization requires a target. Go back to Page 2.")
        st.stop()

    y = df[target_col]
    task = infer_task(y)
    test_size = st.slider("Test fraction", 0.05, 0.5, 0.2)
    rs = st.number_input("Random state", value=42)
    try:
        X_full = preprocessor.fit_transform(df.drop(columns=[target_col]))
    except Exception:
        X_full = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).fillna(0).values

    try:
        X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=test_size, random_state=rs, stratify=(y if task == 'classification' else None))
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=test_size, random_state=rs)

    if task == "classification":
        algs = list(SUPERVISED_CLASSIFIERS.keys())
    else:
        algs = list(SUPERVISED_REGRESSORS.keys())
    alg_choice = st.selectbox("Choose algorithm to optimize", algs)
    st.write("Algorithm:", alg_choice)

    DEFAULT_PARAM_GRIDS = {
        "Logistic Regression": {'clf__C': [0.01, 0.1, 1, 10]},
        "K-Nearest Neighbors": {'clf__n_neighbors': [3,5,7]},
        "SVM (RBF)": {'clf__C': [0.1,1,10], 'clf__gamma': ['scale','auto']},
        "Decision Tree": {'clf__max_depth': [None, 5, 10]},
        "Random Forest": {'clf__n_estimators': [50,100], 'clf__max_depth': [None,5,10]},
        "Gradient Boosting": {'clf__n_estimators': [50,100], 'clf__learning_rate': [0.01,0.1]},
        "Ridge": {'clf__alpha': [0.1,1.0,10]},
        "Lasso": {'clf__alpha': [0.001,0.01,0.1]},
        "SVR": {'clf__C': [0.1,1,10], 'clf__gamma': ['scale','auto']}
    }
    default_grid = DEFAULT_PARAM_GRIDS.get(alg_choice, {})
    st.write("Default param grid (editable):")
    user_grid_text = st.text_area("Param grid as Python dict (keys like 'clf__n_estimators')", value=str(default_grid), height=120)
    try:
        user_grid = eval(user_grid_text) if user_grid_text.strip() else default_grid
        if not isinstance(user_grid, dict):
            user_grid = default_grid
    except Exception:
        user_grid = default_grid

    search_type = st.radio("Search type", ["GridSearchCV", "RandomizedSearchCV"])
    cv = st.number_input("CV folds", min_value=2, max_value=10, value=5)
    pipeline = Pipeline([("clf", SUPERVISED_CLASSIFIERS.get(alg_choice, SUPERVISED_REGRESSORS.get(alg_choice)))])
    n_iter = None
    if search_type == "RandomizedSearchCV":
        n_iter = st.number_input("n_iter", min_value=1, value=10)
    scoring = st.selectbox("Scoring metric", ['accuracy','f1','roc_auc'] if task=='classification' else ['r2','neg_mean_squared_error'])

    if st.button("Run hyperparameter search"):
        with st.spinner("Searching..."):
            try:
                if search_type == "GridSearchCV":
                    searcher = GridSearchCV(pipeline, user_grid, cv=cv, scoring=scoring, n_jobs=-1, refit=True)
                else:
                    searcher = RandomizedSearchCV(pipeline, user_grid, cv=cv, scoring=scoring, n_iter=int(n_iter), n_jobs=-1, refit=True, random_state=rs)
                searcher.fit(X_train, y_train)
                st.success(f"Search done. Best score: {searcher.best_score_:.4f}")
                st.json(searcher.best_params_)
                best_model = searcher.best_estimator_
                y_pred = best_model.predict(X_test)
                if task == "classification":
                    y_prob = None
                    if hasattr(best_model, "predict_proba"):
                        try:
                            y_prob = best_model.predict_proba(X_test)
                        except Exception:
                            y_prob = None
                    metrics = compute_classification_metrics(y_test, y_pred, y_prob=y_prob)
                    st.json(metrics)
                else:
                    metrics = compute_regression_metrics(y_test, y_pred)
                    st.json(metrics)
                st.session_state.store['optimized_model'] = best_model
                st.session_state.store['optimized_metrics'] = metrics
            except Exception as e:
                st.error(f"Search failed: {e}")

# Page 4: Before vs After Comparison
elif page == "4 — Comparison":
    st.header("Before vs After Comparison")
    baseline = st.session_state.store.get("results", {})
    optimized = st.session_state.store.get("optimized_metrics", None)
    ensemble = st.session_state.store.get("ensemble_metrics", None)
    if not baseline:
        st.info("No baseline models in session. Train on Page 2.")
        st.stop()
    st.subheader("Baseline models")
    for k, v in baseline.items():
        st.write(f"**{k}**"); st.json(v)
    if optimized:
        st.subheader("Optimized model metrics")
        st.json(optimized)
    if ensemble:
        st.subheader("Ensemble metrics")
        st.json(ensemble)
    st.markdown("---")
    st.subheader("Comparison table")
    comp_rows = []
    for name, metrics in baseline.items():
        row = {"model": name}
        row.update(metrics)
        comp_rows.append(row)
    if optimized:
        r = {"model": "optimized_model"}; r.update(optimized); comp_rows.append(r)
    if ensemble:
        r = {"model": "ensemble"}; r.update(ensemble); comp_rows.append(r)
    if comp_rows:
        st.dataframe(pd.DataFrame(comp_rows).fillna("NA"))
    else:
        st.write("No results to show.")

st.markdown("---")
st.write("Made with ❤️ — Streamlit + scikit-learn. Suggestor samples the data and runs quick CV across candidate models; adjust sample size for speed.")
st.write("Note: This is a basic implementation for demonstration purposes. For production use, consider more robust error handling, security, and optimizations.")
