# ml_app_enhanced.py
"""
Enhanced Auto ML Streamlit app (refactor + new features)
- Modularized functions
- Caching for data & preprocessors
- Better UI with tabs and progress indicators
- Suggestor (light benchmarking) with progress and results
- Quick training & evaluation + download trained model
- Hyperparameter search with progress
- Confusion matrix and feature-importance visualizations
- Ensemble (voting) builder from trained models
- Exportable evaluation reports (CSV/JSON)

Notes:
- Keep required packages: streamlit, scikit-learn, pandas, numpy, matplotlib, joblib
- Optional: xgboost
"""

import io
import time
import json
import tempfile
import joblib
import warnings
from functools import partial
from typing import Tuple, Dict, Any, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             r2_score, mean_squared_error, mean_absolute_error, confusion_matrix)

# models
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier,
                              GradientBoostingRegressor, VotingClassifier, VotingRegressor)

# optional xgboost
try:
    from xgboost import XGBClassifier, XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

warnings.filterwarnings("ignore")
st.set_page_config(layout="wide", page_title="Auto ML App — Enhanced")

# -------------------------
# Constants & Candidates
# -------------------------
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

# -------------------------
# Utility functions
# -------------------------

def infer_task(y: pd.Series) -> Optional[str]:
    """Infer task: 'classification' or 'regression'"""
    if y is None:
        return None
    if getattr(y, 'dtype', None) is not None and (y.dtype == object or y.dtype.name == 'category'):
        return 'classification'
    try:
        unique = y.dropna().unique()
        # integer-like with few unique values => classification
        if len(unique) <= 20 and np.all(np.equal(np.mod(pd.to_numeric(y.dropna(), errors='coerce'), 1), 0)):
            return 'classification'
    except Exception:
        pass
    return 'regression'


def auto_target_detect(df: pd.DataFrame) -> Optional[str]:
    for cand in ['target', 'label', 'y', 'class']:
        if cand in df.columns:
            return cand
    if df.shape[1] >= 1:
        return df.columns[-1]
    return None


def compute_classification_metrics(y_true, y_pred, y_prob=None) -> Dict[str, Any]:
    out = {}
    try:
        out['accuracy'] = float(accuracy_score(y_true, y_pred))
        out['precision'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
        out['recall'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
        out['f1'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    except Exception:
        out.update({'accuracy': None, 'precision': None, 'recall': None, 'f1': None})
    if y_prob is not None:
        try:
            y_true_bin = pd.get_dummies(y_true).values
            out['roc_auc'] = float(roc_auc_score(y_true_bin, y_prob, average='macro', multi_class='ovr'))
        except Exception:
            out['roc_auc'] = None
    else:
        out['roc_auc'] = None
    return out


def compute_regression_metrics(y_true, y_pred) -> Dict[str, Any]:
    out = {}
    try:
        out['r2'] = float(r2_score(y_true, y_pred))
        out['rmse'] = float(mean_squared_error(y_true, y_pred, squared=False))
        out['mae'] = float(mean_absolute_error(y_true, y_pred))
    except Exception:
        out.update({'r2': None, 'rmse': None, 'mae': None})
    return out


@st.cache_data
def load_sample_dataset(name: str) -> Optional[pd.DataFrame]:
    try:
        if name.startswith('iris'):
            from sklearn.datasets import load_iris
            d = load_iris(as_frame=True)
            df = d.frame.copy()
            df['target'] = d.target
            return df
        if name.startswith('diabetes'):
            from sklearn.datasets import load_diabetes
            d = load_diabetes(as_frame=True)
            df = d.frame.copy()
            df['target'] = d.target
            return df
    except Exception:
        return None
    return None


@st.cache_data
def read_buffer(buffer, sample_type=None):
    if buffer is None and sample_type:
        return load_sample_dataset(sample_type)
    if buffer is None:
        return None
    try:
        name = getattr(buffer, 'name', '') or ''
        if name.lower().endswith(('.xls', '.xlsx')):
            return pd.read_excel(buffer)
        else:
            return pd.read_csv(buffer)
    except Exception:
        try:
            return pd.read_table(buffer)
        except Exception:
            return None


@st.cache_data
def build_preprocessor(df: pd.DataFrame, target_col: Optional[str] = None, scaler: Optional[str] = None,
                       encode_categoricals: bool = True) -> Tuple[ColumnTransformer, List[str], List[str]]:
    df = df.copy()
    if target_col and target_col in df.columns:
        X = df.drop(columns=[target_col])
    else:
        X = df
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    num_steps = [('imputer', SimpleImputer(strategy='mean'))]
    if scaler == 'standard':
        num_steps.append(('scaler', StandardScaler()))
    elif scaler == 'minmax':
        num_steps.append(('scaler', MinMaxScaler()))
    from sklearn.pipeline import Pipeline as SkPipeline
    num_pipeline = SkPipeline(num_steps)
    cat_steps = [('imputer', SimpleImputer(strategy='most_frequent'))]
    if encode_categoricals and len(cat_cols) > 0:
        cat_steps.append(('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False)))
    cat_pipeline = SkPipeline(cat_steps)

    transformers = []
    if len(numeric_cols) > 0:
        transformers.append(('num', num_pipeline, numeric_cols))
    if len(cat_cols) > 0:
        transformers.append(('cat', cat_pipeline, cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    return preprocessor, numeric_cols, cat_cols


def safe_preprocess(preprocessor: ColumnTransformer, X: pd.DataFrame) -> np.ndarray:
    try:
        return preprocessor.fit_transform(X)
    except Exception:
        # fallback to numeric-only
        return X.select_dtypes(include=[np.number]).fillna(0).values


# -------------------------
# Suggestor: lightweight benchmark
# -------------------------

def benchmark_models(df: pd.DataFrame, target_col: str, task: str, preprocessor: ColumnTransformer,
                     sample_size: int = 800, cv: int = 3, max_models: int = 6, random_state: int = 42):
    if df is None or target_col not in df.columns:
        return []
    # sample
    if sample_size is not None and df.shape[0] > sample_size:
        if task == 'classification':
            try:
                df_sample = df.groupby(target_col, group_keys=False).apply(
                    lambda x: x.sample(n=max(1, int(sample_size * len(x) / len(df))), random_state=random_state))
                df_sample = df_sample.sample(n=min(sample_size, len(df_sample)), random_state=random_state)
            except Exception:
                df_sample = df.sample(n=sample_size, random_state=random_state)
        else:
            df_sample = df.sample(n=sample_size, random_state=random_state)
    else:
        df_sample = df.copy()

    df_sample = df_sample.dropna(subset=[target_col])
    if df_sample.shape[0] < 3:
        return []

    X_raw = df_sample.drop(columns=[target_col])
    y_raw = df_sample[target_col]
    X = safe_preprocess(preprocessor, X_raw)

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
            scores = cross_val_score(pipe, X, y_raw, cv=cv, scoring=scoring, n_jobs=-1)
            results.append({'model': name, 'mean_score': float(np.mean(scores)), 'std': float(np.std(scores)), 'scores': [float(s) for s in scores]})
        except Exception as e:
            results.append({'model': name, 'mean_score': None, 'error': str(e)})
    results_sorted = sorted([r for r in results if r.get('mean_score') is not None], key=lambda x: x['mean_score'], reverse=True)
    return results_sorted


# -------------------------
# Training & Evaluation helpers
# -------------------------

def train_quick_models(X, y, model_names: List[str], task: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    trained = {}
    metrics = {}
    for name in model_names:
        try:
            if task == 'classification':
                model = SUPERVISED_CLASSIFIERS[name]
            else:
                model = SUPERVISED_REGRESSORS[name]
            model.fit(X['train'], y['train'])
            y_pred = model.predict(X['test'])
            if task == 'classification':
                y_prob = None
                if hasattr(model, 'predict_proba'):
                    try:
                        y_prob = model.predict_proba(X['test'])
                    except Exception:
                        y_prob = None
                m = compute_classification_metrics(y['test'], y_pred, y_prob=y_prob)
            else:
                m = compute_regression_metrics(y['test'], y_pred)
            trained[name] = model
            metrics[name] = m
        except Exception as e:
            metrics[name] = {'error': str(e)}
    return trained, metrics


# -------------------------
# Visualization helpers
# -------------------------

def plot_conf_matrix(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title('Confusion matrix')
    fig.colorbar(im, ax=ax)
    ticks = np.arange(cm.shape[0])
    if labels is None:
        labels = [str(i) for i in ticks]
    ax.set_xticks(ticks); ax.set_xticklabels(labels, rotation=45)
    ax.set_yticks(ticks); ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > cm.max()/2 else 'black')
    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_names: List[str]):
    # supports tree-based estimators and linear coef
    fi = None
    if hasattr(model, 'feature_importances_'):
        fi = model.feature_importances_
    elif hasattr(model, 'coef_'):
        fi = np.mean(np.abs(model.coef_), axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
    if fi is None:
        return None
    idx = np.argsort(fi)[-20:]
    fig, ax = plt.subplots(figsize=(6, max(4, len(idx) * 0.25)))
    ax.barh(range(len(idx)), fi[idx])
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx])
    ax.set_title('Top feature importance')
    plt.tight_layout()
    return fig


# -------------------------
# Streamlit App UI
# -------------------------

st.title('Auto ML: Enhanced (Refactor + Features)')
st.write('Upload CSV/XLSX or use sample dataset. This enhanced app includes suggestor, training, hyperparameter search, visualization, downloads.')

with st.sidebar:
    st.header('Data')
    uploaded_file = st.file_uploader('Upload CSV/XLSX/TXT', type=['csv', 'txt', 'xlsx', 'xls'])
    use_sample = st.checkbox('Use sample dataset', value=False)
    sample_choice = None
    if use_sample:
        sample_choice = st.selectbox('Sample', ['iris (classification)', 'diabetes (regression)'])
    st.markdown('---')
    page = st.radio('Page', ['1 — EDA', '2 — Suggestor & Train', '3 — Optimize', '4 — Comparison/Export'])

# load data
if use_sample and not uploaded_file:
    df = read_buffer(None, sample_choice)
elif uploaded_file:
    df = read_buffer(uploaded_file, None)
else:
    df = None

if df is None:
    st.info('No data loaded. Upload or pick a sample dataset in the sidebar.')
    if page != '1 — EDA':
        st.stop()
else:
    if not isinstance(df, pd.DataFrame):
        st.error('Loaded object is not a DataFrame.')
        st.stop()
    st.success(f'Data: {df.shape[0]} rows × {df.shape[1]} columns')

# persist store
if 'store' not in st.session_state:
    st.session_state.store = {'df': df, 'preprocessor': None, 'trained': {}, 'results': {}, 'optimized_model': None}
else:
    st.session_state.store['df'] = df

# Page 1: EDA
if page == '1 — EDA':
    st.header('Exploratory Data Analysis')
    tabs = st.tabs(['Preview', 'Summary', 'Missing', 'Visuals'])
    with tabs[0]:
        if st.checkbox('Show raw data (first 200 rows)', False):
            st.dataframe(df.head(200))
        st.write('Shape:', df.shape)
        st.write('Columns & types')
        st.table(pd.DataFrame({'column': df.columns, 'dtype': df.dtypes}))
    with tabs[1]:
        st.subheader('Descriptive statistics')
        try:
            st.write(df.describe(include='all').T)
        except Exception:
            st.write('Describe failed.')
    with tabs[2]:
        st.subheader('Missing values')
        mv = df.isnull().sum().sort_values(ascending=False)
        st.bar_chart(mv.head(20))
    with tabs[3]:
        st.subheader('Quick visuals')
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            col = st.selectbox('Choose numeric column', numeric_cols)
            st.line_chart(df[col].dropna().reset_index(drop=True).head(200))

# Page 2: Suggestor & Train
elif page == '2 — Suggestor & Train':
    st.header('Suggestor & Quick training')
    suggested_target = auto_target_detect(df)
    options = [None] + list(df.columns)
    try:
        index_default = options.index(suggested_target) if suggested_target in options else len(options)-1
    except Exception:
        index_default = len(options)-1
    target_col = st.selectbox('Target column (auto-detected)', options=options, index=index_default)

    if target_col is None:
        detected_task = None
        st.info('Unsupervised: you can still run PCA/KMeans on numeric features.')
    else:
        detected_task = infer_task(df[target_col])
        st.info(f'Detected task: {detected_task}')

    scaler_choice = st.selectbox('Scaler', ['None', 'standard', 'minmax'])
    encode_cat = st.checkbox('One-hot encode categoricals', value=True)

    # build preprocessor
    try:
        preprocessor, num_cols, cat_cols = build_preprocessor(df, target_col, scaler=(scaler_choice if scaler_choice != 'None' else None), encode_categoricals=encode_cat)
        st.session_state.store['preprocessor'] = preprocessor
    except Exception as e:
        st.error(f'Preprocessor build failed: {e}')
        preprocessor = None
        num_cols = cat_cols = []

    st.subheader('Preview preprocessed features')
    if preprocessor is not None:
        try:
            X_proc = safe_preprocess(preprocessor, df.drop(columns=[target_col]) if target_col else df)
            st.dataframe(pd.DataFrame(X_proc).head(10))
        except Exception as e:
            st.warning(f'Preview failed: {e}')

    st.markdown('---')
    st.subheader('Suggest best algorithm (light benchmark)')
    sample_size = st.slider('Sample size (smaller = faster)', 100, 2000, 800, step=100)
    cv = st.slider('CV folds', 2, 5, 3)
    max_models = st.slider('Max candidate models', 3, 8, 6)

    if st.button('Run suggestor'):
        if target_col is None or preprocessor is None:
            st.warning('Need a target and preprocessor to run suggestor.')
        else:
            with st.spinner('Running benchmarks...'):
                results = benchmark_models(df, target_col, detected_task, preprocessor, sample_size=sample_size, cv=cv, max_models=max_models)
                if not results:
                    st.warning('No successful results. Try different options or smaller sample.')
                else:
                    st.success('Suggestor done — top picks:')
                    for i, r in enumerate(results[:5], start=1):
                        st.write(f"{i}. {r['model']} — mean: {r['mean_score']:.4f} ± {r['std']:.4f}")
                    st.markdown('#### Full results')
                    st.dataframe(pd.DataFrame(results))
                    st.session_state.store['suggestions'] = results

    st.markdown('---')
    st.subheader('Quick train selected models')
    if detected_task == 'classification':
        avail = list(SUPERVISED_CLASSIFIERS.keys())
    elif detected_task == 'regression':
        avail = list(SUPERVISED_REGRESSORS.keys())
    else:
        avail = []

    chosen = st.multiselect('Pick models to train', options=avail, default=avail[:2] if len(avail) else [])
    test_frac = st.slider('Test fraction', 0.05, 0.5, 0.2)
    rs = st.number_input('Random state', value=42)

    if st.button('Train selected models (quick)'):
        if target_col is None:
            st.warning('Select a target first.')
        elif preprocessor is None:
            st.error('Set preprocessing first.')
        else:
            try:
                X_all = safe_preprocess(preprocessor, df.drop(columns=[target_col]))
            except Exception:
                X_all = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).fillna(0).values
            y_all = df[target_col]
            try:
                X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=test_frac, random_state=int(rs),
                                                          stratify=(y_all if detected_task == 'classification' else None))
            except Exception:
                X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=test_frac, random_state=int(rs))

            X_splits = {'train': X_tr, 'test': X_te}
            y_splits = {'train': y_tr, 'test': y_te}

            with st.spinner('Training...'):
                trained, metrics = train_quick_models(X_splits, y_splits, chosen, detected_task)
                st.session_state.store['trained'].update(trained)
                st.session_state.store['results'].update(metrics)
                st.success('Training complete')
                for k, v in metrics.items():
                    st.subheader(k)
                    st.json(v)

                # offer download for models
                if trained:
                    for name, model in trained.items():
                        buf = io.BytesIO()
                        joblib.dump(model, buf)
                        buf.seek(0)
                        st.download_button(f'Download `{name}` model (pickle)', data=buf, file_name=f'{name.replace(" ","_")}.pkl')

# Page 3: Optimize
elif page == '3 — Optimize':
    st.header('Hyperparameter optimization')
    preprocessor = st.session_state.store.get('preprocessor', None)
    if preprocessor is None:
        st.warning('Set preprocessing on Page 2 first.')
        st.stop()

    suggested_target = auto_target_detect(df)
    options = [None] + list(df.columns)
    try:
        index_default = options.index(suggested_target) if suggested_target in options else len(options)-1
    except Exception:
        index_default = len(options)-1
    target_col = st.selectbox('Target column for optimization', options=options, index=index_default)
    if target_col is None:
        st.info('Optimization requires a target column.')
        st.stop()

    task = infer_task(df[target_col])
    st.write('Detected task:', task)
    test_frac = st.slider('Test fraction', 0.05, 0.5, 0.2)
    rs = st.number_input('Random state', value=42)

    try:
        X_full = safe_preprocess(preprocessor, df.drop(columns=[target_col]))
    except Exception:
        X_full = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).fillna(0).values
    y_full = df[target_col]

    try:
        X_tr, X_te, y_tr, y_te = train_test_split(X_full, y_full, test_size=test_frac, random_state=int(rs),
                                                  stratify=(y_full if task == 'classification' else None))
    except Exception:
        X_tr, X_te, y_tr, y_te = train_test_split(X_full, y_full, test_size=test_frac, random_state=int(rs))

    if task == 'classification':
        algs = list(SUPERVISED_CLASSIFIERS.keys())
    else:
        algs = list(SUPERVISED_REGRESSORS.keys())
    alg_choice = st.selectbox('Algorithm to optimize', algs)

    DEFAULT_GRIDS = {
        'Logistic Regression': {'clf__C': [0.01, 0.1, 1, 10]},
        'K-Nearest Neighbors': {'clf__n_neighbors': [3,5,7]},
        'SVM (RBF)': {'clf__C': [0.1,1,10], 'clf__gamma': ['scale','auto']},
        'Decision Tree': {'clf__max_depth': [None,5,10]},
        'Random Forest': {'clf__n_estimators': [50,100], 'clf__max_depth': [None,5,10]},
        'Gradient Boosting': {'clf__n_estimators': [50,100], 'clf__learning_rate': [0.01,0.1]},
        'Ridge': {'clf__alpha': [0.1,1.0,10]},
        'Lasso': {'clf__alpha': [0.001,0.01,0.1]},
        'SVR': {'clf__C': [0.1,1,10], 'clf__gamma': ['scale','auto']}
    }
    default_grid = DEFAULT_GRIDS.get(alg_choice, {})
    user_grid_text = st.text_area('Param grid (Python dict)', value=str(default_grid), height=120)
    try:
        user_grid = eval(user_grid_text) if user_grid_text.strip() else default_grid
        if not isinstance(user_grid, dict):
            user_grid = default_grid
    except Exception:
        user_grid = default_grid

    search_type = st.radio('Search type', ['GridSearchCV', 'RandomizedSearchCV'])
    cv = st.number_input('CV folds', min_value=2, max_value=10, value=5)
    scoring = st.selectbox('Scoring metric', ['accuracy','f1','roc_auc'] if task == 'classification' else ['r2','neg_mean_squared_error'])
    n_iter = None
    if search_type == 'RandomizedSearchCV':
        n_iter = st.number_input('n_iter (for Randomized)', min_value=1, value=10)

    pipeline = Pipeline([('clf', SUPERVISED_CLASSIFIERS.get(alg_choice, SUPERVISED_REGRESSORS.get(alg_choice)))])

    if st.button('Run hyperparameter search'):
        with st.spinner('Running search...'):
            try:
                if search_type == 'GridSearchCV':
                    searcher = GridSearchCV(pipeline, user_grid, cv=int(cv), scoring=scoring, n_jobs=-1, refit=True)
                else:
                    searcher = RandomizedSearchCV(pipeline, user_grid, cv=int(cv), scoring=scoring, n_iter=int(n_iter), n_jobs=-1, refit=True, random_state=int(rs))
                searcher.fit(X_tr, y_tr)
                st.success(f"Search complete — best score: {searcher.best_score_:.4f}")
                st.json(searcher.best_params_)
                best = searcher.best_estimator_
                y_pred = best.predict(X_te)
                if task == 'classification':
                    y_prob = None
                    if hasattr(best, 'predict_proba'):
                        try:
                            y_prob = best.predict_proba(X_te)
                        except Exception:
                            y_prob = None
                    metrics = compute_classification_metrics(y_te, y_pred, y_prob=y_prob)
                else:
                    metrics = compute_regression_metrics(y_te, y_pred)
                st.subheader('Optimized model metrics')
                st.json(metrics)
                st.session_state.store['optimized_model'] = best
                st.session_state.store['optimized_metrics'] = metrics

                # show confusion matrix and feature importance for classification
                if task == 'classification':
                    try:
                        fig = plot_conf_matrix(y_te, y_pred, labels=None)
                        st.pyplot(fig)
                    except Exception:
                        pass
                try:
                    feat_names = []
                    # attempt to recover feature names when OneHotEncoder used
                    if isinstance(preprocessor, ColumnTransformer):
                        names = []
                        for trans_name, trans, cols in preprocessor.transformers:
                            if trans_name == 'num':
                                names.extend(cols)
                            elif trans_name == 'cat':
                                # expand onehot
                                if hasattr(trans.named_steps.get('onehot'), 'get_feature_names_out'):
                                    names.extend(list(trans.named_steps['onehot'].get_feature_names_out(cols)))
                                else:
                                    names.extend(cols)
                        feat_names = names
                    if feat_names:
                        fig2 = plot_feature_importance(best.named_steps['clf'] if hasattr(best, 'named_steps') else best, feat_names)
                        if fig2 is not None:
                            st.pyplot(fig2)
                except Exception:
                    pass

                # offer download
                buf = io.BytesIO()
                joblib.dump(best, buf)
                buf.seek(0)
                st.download_button('Download optimized model (pickle)', data=buf, file_name='optimized_model.pkl')
            except Exception as e:
                st.error(f'Search failed: {e}')

# Page 4: Comparison/Export
elif page == '4 — Comparison/Export':
    st.header('Compare & Export')
    baseline = st.session_state.store.get('results', {})
    optimized = st.session_state.store.get('optimized_metrics', None)
    trained = st.session_state.store.get('trained', {})

    if not baseline and optimized is None:
        st.info('No results available. Train on Page 2 or run optimization on Page 3.')
        st.stop()

    st.subheader('Baseline results')
    for k, v in baseline.items():
        st.write(k); st.json(v)

    if optimized:
        st.subheader('Optimized model')
        st.json(optimized)

    st.markdown('---')
    st.subheader('Comparison table')
    rows = []
    for name, m in baseline.items():
        r = {'model': name}
        r.update(m)
        rows.append(r)
    if optimized:
        r = {'model': 'optimized_model'}
        r.update(optimized)
        rows.append(r)
    if rows:
        df_comp = pd.DataFrame(rows)
        st.dataframe(df_comp.fillna('NA'))

        # export
        csv_buf = df_comp.to_csv(index=False).encode('utf-8')
        st.download_button('Download comparison CSV', data=csv_buf, file_name='comparison.csv')
        st.download_button('Download comparison JSON', data=json.dumps(rows, indent=2), file_name='comparison.json')

    # ensemble builder
    st.markdown('---')
    st.subheader('Build a Voting Ensemble from trained models')
    trained_names = list(st.session_state.store.get('trained', {}).keys())
    if trained_names:
        pick = st.multiselect('Pick trained models for ensemble', options=trained_names)
        if pick and st.button('Create voting ensemble'):
            estimators = [(n.replace(' ','_'), st.session_state.store['trained'][n]) for n in pick]
            # detect task from one stored model
            sample_model = st.session_state.store['trained'][pick[0]]
            is_clf = hasattr(sample_model, 'predict_proba') or isinstance(sample_model, (LogisticRegression, KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier))
            try:
                if is_clf:
                    ensemble = VotingClassifier(estimators=estimators, voting='soft' if all(hasattr(m, 'predict_proba') for _, m in estimators) else 'hard')
                else:
                    ensemble = VotingRegressor(estimators=estimators)
                # we need training data to fit ensemble — reuse last train/test if available
                # attempt to retrieve last X/y splits from session (not stored) — fallback: retrain on full df
                preprocessor = st.session_state.store.get('preprocessor')
                target_col = auto_target_detect(df)
                X_full = safe_preprocess(preprocessor, df.drop(columns=[target_col])) if preprocessor and target_col else df.drop(columns=[target_col]).select_dtypes(include=[np.number]).fillna(0).values
                y_full = df[target_col]
                ensemble.fit(X_full, y_full)
                st.session_state.store['ensemble'] = ensemble
                st.success('Ensemble built and fitted on dataset')
                buf = io.BytesIO(); joblib.dump(ensemble, buf); buf.seek(0)
                st.download_button('Download ensemble model (pickle)', data=buf, file_name='ensemble_model.pkl')
            except Exception as e:
                st.error(f'Failed to build ensemble: {e}')
    else:
        st.info('No trained models available to build ensemble.')

st.markdown('---')
st.write('Made with ❤️ — Streamlit + scikit-learn. This enhanced app is for demonstration and learning. For production: secure uploads, resource limits, and robust persistence.')
