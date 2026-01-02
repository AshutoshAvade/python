"""
Streamlit ML Playground - single-file application
Features:
- Upload dataset (CSV)
- 1st page: Exploratory Data Analysis (EDA)
- 2nd page: Algorithms (supervised: classification/regression; unsupervised: clustering/dimensionality reduction)
- 3rd page: Model evaluation, hyperparameter tuning, ensemble learning
- 4th page: Plotting & visualizations (customizable)

How to run:
1. Create a virtual env (recommended)
2. pip install streamlit pandas numpy scikit-learn matplotlib seaborn plotly xgboost lightgbm shap
   (xgboost/lightgbm/shap optional; code gracefully handles missing packages)
3. streamlit run ml_app.py

Notes:
- The app auto-detects target column when possible, but you can set it manually.
- For classification/regression detection, the app uses pandas dtype and number of unique values heuristics.
- For big datasets, sampling is used for quick EDA/plots.

"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import time
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# optional imports
try:
    import xgboost as xgb
    XGBClassifier = xgb.XGBClassifier
    XGBRegressor = xgb.XGBRegressor
except Exception:
    XGBClassifier = None
    XGBRegressor = None

try:
    import lightgbm as lgb
    LGBMClassifier = lgb.LGBMClassifier
    LGBMRegressor = lgb.LGBMRegressor
except Exception:
    LGBMClassifier = None
    LGBMRegressor = None

st.set_page_config(page_title="ML Playground", layout="wide")

# ------------------ Utilities ------------------
@st.cache_data
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

@st.cache_data
def df_sample(df, n=5000):
    if df.shape[0] > n:
        return df.sample(n, random_state=42)
    return df

def infer_task(y: pd.Series):
    # heuristics to detect classification vs regression
    if y.dtype == 'object' or y.dtype.name == 'category':
        return 'classification'
    if y.nunique() < 20 and y.nunique() / len(y) < 0.05:
        # likely classification (small set of categories)
        return 'classification'
    return 'regression'

def quick_summary(df):
    desc = df.describe(include='all').T
    desc['missing'] = df.isnull().sum()
    return desc

# ------------------ Pages ------------------

st.title("ML Playground — Upload, Explore, Train, Visualize")

# Sidebar - file upload and navigation
st.sidebar.header("Dataset & Navigation")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])

page = st.sidebar.radio("Select page", ["1 - EDA", "2 - Algorithms", "3 - Evaluation & Optimization", "4 - Plots"])

if uploaded_file is None:
    st.info("Upload a CSV file to get started. You can also use the sample dataset below.")
    if st.button("Use sample: Iris & California Housing combined chooser"):
        sample_choice = st.selectbox("Choose sample dataset", ["Iris (classification)", "California housing (regression)"])
        if sample_choice == "Iris (classification)":
            from sklearn.datasets import load_iris
            data = load_iris(as_frame=True)
            df = data.frame
        else:
            from sklearn.datasets import fetch_california_housing
            data = fetch_california_housing(as_frame=True)
            df = pd.concat([data.frame, pd.Series(data.target, name='target')], axis=1)
else:
    df = load_csv(uploaded_file)

if 'df' not in locals():
    st.stop()

# Basic info
with st.expander("Dataset info"):
    st.write(f"Shape: {df.shape}")
    st.dataframe(df.head())

# Sidebar: target selection & task detection
st.sidebar.subheader("Target / Task")
all_cols = df.columns.tolist()
default_target = None
if 'target' in df.columns:
    default_target = 'target'
elif len(df.columns) >= 2:
    default_target = df.columns[-1]

target_col = st.sidebar.selectbox("Select target column (for supervised tasks)", options=[None] + all_cols, index=0 if default_target is None else (all_cols.index(default_target) + 1))

# If target selected, infer task
task = None
if target_col:
    task = infer_task(df[target_col])
    task_choice = st.sidebar.radio("Detected task", [task, 'classification' if task=='regression' else 'regression'], index=0)
    task = task_choice
else:
    st.sidebar.write("No target selected — unsupervised pages will be enabled.")

# Common preprocessing helper
def preprocess(df, target=None, numeric_strategy='mean', categorical_strategy='most_frequent'):
    X = df.drop(columns=[target]) if target else df.copy()
    y = df[target] if target else None
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=numeric_strategy)),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=categorical_strategy)),
        ('encode',  # simple label encoding in transformer using function
         'passthrough')
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', 'passthrough', categorical_features)
        ])
    return X, y, preprocessor, numeric_features, categorical_features

# ---------- Page 1: EDA ----------
if page.startswith('1'):
    st.header("Exploratory Data Analysis (EDA)")
    st.markdown("Use the panels below to inspect distribution, missing values, correlations, and sample rows.")

    with st.expander("Summary & Missing values"):
        st.write(quick_summary(df))
        missing = df.isnull().sum().sort_values(ascending=False)
        st.bar_chart(missing[missing>0])

    with st.expander("Data types & Unique values"):
        dtypes = pd.DataFrame(df.dtypes, columns=['dtype'])
        dtypes['n_unique'] = df.nunique()
        st.dataframe(dtypes)

    with st.expander("Value counts for categorical columns"):
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        for c in cat_cols:
            st.write(f"**{c}**")
            st.write(df[c].value_counts().head(20))

    with st.expander("Distribution plots (numeric)"):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        sel = st.multiselect("Select numeric columns to plot", numeric_cols, default=numeric_cols[:3])
        sample = df_sample(df)
        for c in sel:
            fig, ax = plt.subplots()
            sns.histplot(sample[c].dropna(), kde=True, ax=ax)
            ax.set_title(c)
            st.pyplot(fig)

    with st.expander("Correlation matrix"):
        corr = df.select_dtypes(include=[np.number]).corr()
        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    with st.expander("Pairplot sample (slow on large datasets)"):
        cols = st.multiselect("Columns for pairplot (max 6)", numeric_cols, default=numeric_cols[:4])
        if len(cols) > 1:
            sample = df_sample(df[cols].dropna(), n=1000)
            fig = sns.pairplot(sample)
            st.pyplot(fig)

# ---------- Page 2: Algorithms ----------
elif page.startswith('2'):
    st.header("Algorithms — Train & Inspect Models")
    st.write("Choose algorithm family and train. For supervised tasks, select the target column in the sidebar.")

    mode = st.radio("Mode", ['Supervised', 'Unsupervised'])

    if mode == 'Supervised' and not target_col:
        st.warning("Select a target column in the sidebar to use supervised mode.")
        st.stop()

    if mode == 'Supervised':
        st.subheader("Supervised Algorithms")
        problem_type = st.selectbox("Problem type", ['Auto-detect', 'Classification', 'Regression'], index=0)
        if problem_type != 'Auto-detect':
            problem_type = problem_type.lower()
        else:
            problem_type = task

        train_frac = st.slider("Train fraction", 0.1, 0.9, 0.75)

        X, y, preprocessor, num_cols, cat_cols = preprocess(df, target=target_col)
        # simple label encoding if categorical y
        if y is not None and y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_frac, random_state=42, stratify=y if problem_type=='classification' else None)

        classifiers = {
            'Logistic Regression': LogisticRegression(max_iter=500),
            'Random Forest': RandomForestClassifier(n_estimators=100),
            'SVC': SVC(probability=True),
            'KNN': KNeighborsClassifier(),
        }
        regressors = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100),
            'SVR': SVR(),
            'KNN': KNeighborsRegressor(),
        }
        if XGBClassifier:
            classifiers['XGBoost'] = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        if XGBRegressor:
            regressors['XGBoost'] = XGBRegressor()
        if LGBMClassifier:
            classifiers['LightGBM'] = LGBMClassifier()
        if LGBMRegressor:
            regressors['LightGBM'] = LGBMRegressor()

        if problem_type == 'classification':
            model_choice = st.selectbox("Choose classifier", list(classifiers.keys()))
            model = classifiers[model_choice]
        else:
            model_choice = st.selectbox("Choose regressor", list(regressors.keys()))
            model = regressors[model_choice]

        if st.button("Train model"):
            with st.spinner('Training...'):
                # build pipeline
                numeric_features = num_cols
                categorical_features = cat_cols
                # for simplicity: fill categorical with most_frequent and label encode outside
                X_train_proc = X_train.copy()
                X_test_proc = X_test.copy()
                for c in categorical_features:
                    X_train_proc[c] = X_train_proc[c].fillna('missing')
                    X_test_proc[c] = X_test_proc[c].fillna('missing')
                    lecol = LabelEncoder()
                    X_train_proc[c] = lecol.fit_transform(X_train_proc[c].astype(str))
                    X_test_proc[c] = lecol.transform(X_test_proc[c].astype(str))

                # scale numeric
                scaler = StandardScaler()
                X_train_proc[numeric_features] = scaler.fit_transform(X_train_proc[numeric_features])
                X_test_proc[numeric_features] = scaler.transform(X_test_proc[numeric_features])

                model.fit(X_train_proc, y_train)

                y_pred = model.predict(X_test_proc)
                st.success('Training completed')

                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Model**")
                    st.write(model)
                with col2:
                    st.write("**Performance (test)**")
                    if problem_type == 'classification':
                        st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
                        st.write(f"F1: {f1_score(y_test, y_pred, average='weighted'):.4f}")
                        if hasattr(model, 'predict_proba'):
                            proba = model.predict_proba(X_test_proc)
                            try:
                                st.write(f"ROC AUC (macro): {roc_auc_score(y_test, proba, multi_class='ovo', average='macro'):.4f}")
                            except Exception:
                                pass
                    else:
                        st.write(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.4f}")
                        st.write(f"R2: {r2_score(y_test, y_pred):.4f}")

                # feature importance if available
                if hasattr(model, 'feature_importances_'):
                    try:
                        fi = model.feature_importances_
                        feat_names = numeric_features + categorical_features
                        fi_df = pd.DataFrame({'feature': feat_names, 'importance': fi})
                        fi_df = fi_df.sort_values('importance', ascending=False).head(20)
                        fig, ax = plt.subplots()
                        sns.barplot(x='importance', y='feature', data=fi_df, ax=ax)
                        st.pyplot(fig)
                    except Exception:
                        pass

    else:
        st.subheader("Unsupervised Algorithms")
        unsup_choice = st.selectbox("Algorithm", ['KMeans', 'DBSCAN', 'PCA (dimensionality reduction)'])
        n_components = st.slider("n components / clusters", 1, min(10, df.shape[1]), value=2)
        sampled = df_sample(df.select_dtypes(include=[np.number]).dropna(), n=5000)
        if sampled.empty:
            st.warning('No numeric columns available for unsupervised methods')
        else:
            Xu = sampled.values
            if unsup_choice == 'KMeans':
                kmeans = KMeans(n_clusters=n_components)
                labels = kmeans.fit_predict(Xu)
                st.write('Cluster counts:')
                st.write(pd.Series(labels).value_counts())
                fig, ax = plt.subplots()
                if Xu.shape[1] >= 2:
                    ax.scatter(Xu[:,0], Xu[:,1], c=labels, alpha=0.6)
                    ax.set_xlabel('feature 0')
                    ax.set_ylabel('feature 1')
                st.pyplot(fig)
            elif unsup_choice == 'DBSCAN':
                db = DBSCAN()
                labels = db.fit_predict(Xu)
                st.write('Cluster counts (including -1 for noise):')
                st.write(pd.Series(labels).value_counts())
            else:
                pca = PCA(n_components=n_components)
                transformed = pca.fit_transform(Xu)
                st.write('Explained variance ratio:')
                st.write(pca.explained_variance_ratio_)
                fig, ax = plt.subplots()
                if n_components >= 2:
                    ax.scatter(transformed[:,0], transformed[:,1], alpha=0.6)
                st.pyplot(fig)

# ---------- Page 3: Evaluation & Optimization ----------
elif page.startswith('3'):
    st.header("Model Evaluation, Optimization & Ensembles")
    st.write("Train models, run GridSearchCV, cross-validation and create ensemble models.")

    if not target_col:
        st.warning('Select a target column for supervised workflows in the sidebar.')
        st.stop()

    X, y, preprocessor, num_cols, cat_cols = preprocess(df, target=target_col)
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    task = infer_task(pd.Series(y))
    st.write(f"Detected task: **{task}**")

    test_size = st.slider('Test size', 0.05, 0.5, 0.25)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y if task=='classification' else None)

    # choose a list of candidate estimators
    if task == 'classification':
        estimators = [
            ('lr', LogisticRegression(max_iter=500)),
            ('rf', RandomForestClassifier(n_estimators=100)),
            ('knn', KNeighborsClassifier()),
        ]
        if XGBClassifier:
            estimators.append(('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss')))
        if LGBMClassifier:
            estimators.append(('lgbm', LGBMClassifier()))
    else:
        estimators = [
            ('lr', LinearRegression()),
            ('rf', RandomForestRegressor(n_estimators=100)),
            ('knn', KNeighborsRegressor()),
        ]
        if XGBRegressor:
            estimators.append(('xgb', XGBRegressor()))
        if LGBMRegressor:
            estimators.append(('lgbm', LGBMRegressor()))

    st.subheader('Cross-validation scores (default 5-fold)')
    cv_folds = st.slider('CV folds', 2, 10, 5)
    scoring = st.selectbox('Scoring metric', ('accuracy' if task=='classification' else 'neg_mean_squared_error'), index=0)

    if st.button('Run cross-val for all estimators'):
        results = {}
        for name, est in estimators:
            pipe = Pipeline([('pre', preprocessor), ('model', est)])
            try:
                scores = cross_val_score(pipe, X, y, cv=cv_folds, scoring=scoring)
                results[name] = scores
                st.write(f"{name}: mean={np.mean(scores):.4f}, std={np.std(scores):.4f}")
            except Exception as e:
                st.write(f"{name}: failed: {e}")

    st.subheader('Grid Search / Hyperparameter Tuning')
    chosen = st.multiselect('Pick estimators to tune', [name for name,_ in estimators])
    param_grid_text = st.text_area('Enter simple param grid in python dict form per estimator name, example:\n{"rf": {"n_estimators": [50,100]}, "knn": {"n_neighbors": [3,5,7]}}', value='')

    if st.button('Run GridSearch'):
        if not param_grid_text:
            st.error('Provide param grid text to run GridSearch')
        else:
            try:
                param_grid = eval(param_grid_text)
            except Exception as e:
                st.error(f'Param grid parse error: {e}')
                param_grid = {}
            best_models = {}
            for name, est in estimators:
                if name in chosen and name in param_grid:
                    pipe = Pipeline([('pre', preprocessor), ('model', est)])
                    try:
                        gs = GridSearchCV(pipe, param_grid={f'model__{k}': v for k,v in param_grid[name].items()}, cv=cv_folds, scoring=scoring, n_jobs=-1)
                        gs.fit(X, y)
                        st.write(f'{name} best: {gs.best_score_} with params {gs.best_params_}')
                        best_models[name] = gs.best_estimator_
                    except Exception as e:
                        st.write(f'GridSearch failed for {name}: {e}')

            if best_models:
                st.success('GridSearch finished')

    st.subheader('Ensemble builder')
    st.write('Select base estimators to build Voting / Stacking ensemble')
    base_names = st.multiselect('Base estimators', [name for name,_ in estimators], default=[estimators[0][0]])
    ensemble_type = st.selectbox('Ensemble type', ['Voting', 'Stacking'])
    if st.button('Build & evaluate ensemble'):
        base_list = [(name, dict(estimators)[name]) for name in base_names]
        if ensemble_type == 'Voting':
            if task=='classification':
                ens = VotingClassifier(estimators=base_list, voting='soft')
            else:
                ens = VotingRegressor(estimators=base_list)
        else:
            if task=='classification':
                ens = StackingClassifier(estimators=base_list, final_estimator=LogisticRegression())
            else:
                ens = StackingRegressor(estimators=base_list, final_estimator=LinearRegression())
        pipe = Pipeline([('pre', preprocessor), ('model', ens)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        st.write('Ensemble performance:')
        if task=='classification':
            st.write(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
        else:
            st.write(f'RMSE: {mean_squared_error(y_test, y_pred, squared=False):.4f}')

# ---------- Page 4: Plots ----------
elif page.startswith('4'):
    st.header('Plotting & Custom Visualizations')
    st.write('Create charts from your dataframe: histogram, boxplot, scatter, line, and custom pairplot samples.')

    plot_type = st.selectbox('Chart type', ['Histogram', 'Boxplot', 'Scatter', 'Line', 'Heatmap correlation'])
    if plot_type == 'Histogram':
        col = st.selectbox('Numeric column', df.select_dtypes(include=[np.number]).columns)
        bins = st.slider('Bins', 5, 200, 30)
        fig, ax = plt.subplots()
        ax.hist(df[col].dropna(), bins=bins)
        ax.set_title(col)
        st.pyplot(fig)
    elif plot_type == 'Boxplot':
        col = st.selectbox('Numeric column', df.select_dtypes(include=[np.number]).columns)
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col].dropna(), ax=ax)
        st.pyplot(fig)
    elif plot_type == 'Scatter':
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        xcol = st.selectbox('X axis', num_cols)
        ycol = st.selectbox('Y axis', num_cols, index=1 if len(num_cols)>1 else 0)
        hue = st.selectbox('Color by (optional)', [None] + all_cols)
        fig, ax = plt.subplots()
        if hue and hue in df.columns:
            sns.scatterplot(data=df, x=xcol, y=ycol, hue=hue, ax=ax)
        else:
            ax.scatter(df[xcol], df[ycol], alpha=0.6)
        st.pyplot(fig)
    elif plot_type == 'Line':
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        col = st.selectbox('Numeric column', num_cols)
        fig, ax = plt.subplots()
        ax.plot(df[col].reset_index(drop=True))
        ax.set_title(col)
        st.pyplot(fig)
    else:
        corr = df.select_dtypes(include=[np.number]).corr()
        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
        st.pyplot(fig)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ — modify this single-file Streamlit app to add custom models, visualizations, or extended preprocessing.")
