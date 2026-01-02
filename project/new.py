# =========================================
# PROJECT: Auto ML / EDA / Feature Dashboard
# Fully merged, fixed version with Auto Feature Engineering
# =========================================

import pandas as pd
import numpy as np
import streamlit as st

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
# Try to import SMOTE from imbalanced-learn; provide graceful fallback if unavailable
try:
    import importlib
    imblearn_os = importlib.import_module("imblearn.over_sampling")
    SMOTE = getattr(imblearn_os, "SMOTE", None)
except (ImportError, ModuleNotFoundError):
    SMOTE = None
# Lazy import for umap will be performed where UMAP is used to avoid import errors
umap = None

# ===============================================================
# UTILITY FUNCTIONS
# ===============================================================

def create_preprocessor(df, apply_feature_engineering=False):
    """Automatically builds preprocessing pipeline with optional feature engineering"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    if apply_feature_engineering:
        numeric_pipeline.steps.append(
            ("poly", PolynomialFeatures(degree=2, include_bias=False, interaction_only=False))
        )

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("rare_group", RareCategoryGrouper(threshold=0.01)),  # Custom rare category handling
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

    return preprocessor, categorical_cols, numeric_cols

class RareCategoryGrouper:
    """Custom transformer to group rare categories into 'Other'"""
    def __init__(self, threshold=0.01):
        self.threshold = threshold
        self.mappings = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        for col in X.columns:
            freq = X[col].value_counts(normalize=True)
            rare_labels = freq[freq < self.threshold].index.tolist()
            self.mappings[col] = rare_labels
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            rare_labels = self.mappings.get(col, [])
            X[col] = X[col].apply(lambda x: 'Other' if x in rare_labels else x)
        return X

def transform_to_df(preprocessor, X_raw):
    """Safely transform dataframe using preprocessor, auto-fit if needed"""
    from sklearn.exceptions import NotFittedError
    try:
        X_np = preprocessor.transform(X_raw)
    except NotFittedError:
        preprocessor.fit(X_raw)
        X_np = preprocessor.transform(X_raw)

    try:
        feature_names = preprocessor.get_feature_names_out()
    except:
        feature_names = [f"f{i}" for i in range(X_np.shape[1])]

    return pd.DataFrame(X_np, columns=feature_names)

def detect_outliers(df, z_thresh=3):
    """Returns indices of outliers using z-score"""
    from scipy.stats import zscore
    numeric_cols = df.select_dtypes(include=[np.number])
def balance_data_smote(X, y):
    # If imblearn's SMOTE is available, use it.
    try:
        if SMOTE is not None:
            sm = SMOTE(random_state=42)
            X_res, y_res = sm.fit_resample(X, y)
            return X_res, y_res
    except Exception:
        # fall through to fallback
        pass

    # Fallback: simple random oversampling to match majority class (works without imblearn)
    from sklearn.utils import resample

    # Ensure X and y are pandas objects for easy concatenation
    X_df = X.copy() if hasattr(X, "copy") else pd.DataFrame(X)
    y_ser = pd.Series(y).reset_index(drop=True)
    X_df = X_df.reset_index(drop=True)

    class_counts = y_ser.value_counts()
    max_count = class_counts.max()

    dfs = []
    ys = []
    for cls, count in class_counts.items():
        cls_idx = y_ser[y_ser == cls].index
        X_cls = X_df.loc[cls_idx]
        y_cls = y_ser.loc[cls_idx]
        if count < max_count:
            X_upsampled, y_upsampled = resample(X_cls, y_cls, replace=True, n_samples=max_count, random_state=42)
        else:
            X_upsampled, y_upsampled = X_cls, y_cls
        dfs.append(X_upsampled)
        ys.append(y_upsampled)

    X_res = pd.concat(dfs).reset_index(drop=True)
    y_res = pd.concat(ys).reset_index(drop=True)

    return X_res, y_res
def balance_data_smote(X, y):
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res

# ===============================================================
# STREAMLIT APP START
# ===============================================================

st.title("🤖 Auto ML Dashboard with Smart Preprocessing & Feature Engineering")

# -------------------------------
# Page 1: Load Data
# -------------------------------
st.subheader("📂 Upload Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    st.write("Raw Dataset Preview", df_raw.head())
    target_column = st.selectbox("Select Target Column", df_raw.columns)
    y_raw = df_raw[target_column]
    X_raw = df_raw.drop(columns=[target_column])
else:
    st.stop()

# -------------------------------
# Page 2: Preprocessing Pipeline
# -------------------------------
st.subheader("🔧 Build Preprocessing Pipeline")

apply_fe = st.checkbox("Apply Automatic Feature Engineering (Polynomial & Interactions)")

if st.button("Create Preprocessing Pipeline"):
    preprocessor, cat_cols, num_cols = create_preprocessor(X_raw, apply_feature_engineering=apply_fe)

    try:
        preprocessor.fit(X_raw)
        st.session_state.preprocessor = preprocessor
        st.success("✅ Preprocessor created & fitted successfully!")

    except Exception as e:
        st.error(f"Failed to fit preprocessor: {e}")

    st.write("Categorical Columns:", cat_cols)
    st.write("Numerical Columns:", num_cols)

# -------------------------------
# Page 3: Transform Data
# -------------------------------
st.subheader("⚡ Transform Data")

if "preprocessor" in st.session_state:
    X = transform_to_df(st.session_state.preprocessor, X_raw)
    st.write("Transformed Feature Data", X.head())
else:
    st.warning("Please create preprocessing pipeline first.")

# -------------------------------
# Page 4: Outlier Detection
# -------------------------------
st.subheader("🚨 Detect Outliers")

if st.button("Detect Outliers"):
    outliers_idx = detect_outliers(X)
    st.write(f"Detected {len(outliers_idx)} outlier rows")
    st.write(list(outliers_idx)[:10])  # Show first 10 indices

# -------------------------------
# Page 5: Class Imbalance & Balancing
# -------------------------------
st.subheader("⚖️ Class Imbalance & Balancing")

if st.button("Check & Balance Data"):
    class_counts = y_raw.value_counts()
    st.write("Class Distribution Before Balancing:", class_counts.to_dict())

    if class_counts.min() / class_counts.max() < 0.5:
        st.info("⚠️ Imbalanced detected. Applying SMOTE...")
        X_bal, y_bal = balance_data_smote(X, y_raw)
        st.write("Balanced Dataset Class Distribution:", pd.Series(y_bal).value_counts().to_dict())
    else:
        st.success("Data is balanced. No action needed.")

# -------------------------------
# Page 6: UMAP Visualization
# -------------------------------
st.subheader("🗺️ UMAP Visualization")
if st.button("Run UMAP"):
    # Lazy import to avoid top-level import errors in environments without umap-learn
    try:
        import umap as _umap
    except Exception:
        try:
            import umap.umap_ as _umap
        except Exception:
            st.error("UMAP (umap-learn) is not installed; install via `pip install umap-learn` to use this feature.")
            st.stop()

    reducer = _umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(X)
    st.write("UMAP Embedding Shape:", embedding.shape)
    umap_df = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"])
    umap_df["target"] = y_raw.values
    st.write(umap_df.head())
    st.write(umap_df.head())

# -------------------------------
# Page 7: Feature Importance (RandomForest)
# -------------------------------
st.subheader("🌟 Feature Importance")

if st.button("Run RandomForest Feature Importance"):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y_raw)
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)
    st.write(importances.head(20))  # Top 20 features

# -------------------------------
# Page 8: Auto Hyperparameter Tuning
# -------------------------------
st.subheader("⚙️ Auto Hyperparameter Tuning (RandomForest)")

if st.button("Run Auto Tuning"):
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5, 10]
    }
    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1)
    grid.fit(X, y_raw)
    st.write("Best Hyperparameters:", grid.best_params_)
    st.write("Best CV Score:", grid.best_score_)
