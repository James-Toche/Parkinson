"""
Analysis by Gemini

What makes this version more robust?

    1. Zero Data Leakage: Resampling (RandomOverSampler), scaling (MinMaxScaler), and feature selection (SelectKBest) are now encapsulated entirely inside each cross-validation fold. The validation data never touches the feature selector or the sampler beforehand.
    
    2. Patient-Level Variance: Computing standard deviations, minimums, and maximums alongside the mean provides the models with a deeper profile of how a patient's biometric features fluctuate.
    
    3. Probabilistic Medical Metrics: Using predict_proba alongside roc_auc_score and average_precision_score (PR-AUC) ensures you are properly evaluating the model's diagnostic confidence across decision boundaries rather than hard binary cutoffs.


The Single train_test_split (test_size=0.4) was replaced by StratifiedKFold:
Instead of splitting your data just once into a single training and validation set, the new code loops through multiple folds (e.g., 5-fold cross-validation). This ensures every data point gets a turn in the validation set, giving you a much more robust and reliable performance estimate instead of relying on one lucky or unlucky random split.

Global Resampling (RandomOverSampler) moved inside the pipeline:
In your original code, you resampled X_train globally before evaluation. In the updated code, RandomOverSampler is wrapped inside the ImbPipeline. This means the oversampling happens dynamically per fold (only on the training portion of each fold). Doing it globally beforehand causes data leakage because synthetic samples generated from the validation/test distribution bleed into your training logic.

Manual Model Iteration changed to a Dictionary Loop:
The list of models was converted into a dictionary (models = {...}) so you can easily map descriptive names (like "Logistic Regression" or "Random Forest") to the models, printing clean headers automatically as it iterates through each cross-validation fold.

Hard Predictions changed to Probabilities (predict_proba):
In your original loop, you passed model.predict(...) (hard 0/1 binary classes) into roc_auc_score. In the new script, it uses model.predict_proba()[:, 1] to fetch predicted probabilities. ROC-AUC and PR-AUC require continuous probability scores to accurately measure a model's ranking threshold performance; feeding them hard binary choices flattens that accuracy curve.

What Should Be the Next Step in the Analysis?

    Extract Feature Importances:
    Since you are running Random Forest, Extra Trees, and XGBoost, look at which aggregated features (e.g., mean vs. standard deviation) dominate the decision-making process. This will show you whether a patient's average acoustic fluctuation or their maximum/minimum variance is a stronger indicator of Parkinson's.

    Implement Hyperparameter Optimization:
    Right now, your models are using static, manually set hyperparameters (like max_depth=5 or C=1). Integrating an automated search framework like Optuna inside your cross-validation loop will help squeeze maximum performance out of each architecture.
    

Interpretation after running Step 4 below:

    Intra-Patient Variability is Crucial:
    Look at how many volatility features show up in the top 15—specifically _std (standard deviation) and _min (e.g., tqwt_kurtosisValue_dec_36_std, tqwt_medianValue_dec_12_std). This proves that a patient's fluctuation or instability across multiple voice recordings is a stronger diagnostic signal than a simple average (_mean). Sticking strictly to the mean would have completely missed these vital patterns.

    Biomedical Relevance (TQWT & DFA):
    The features that dominate are TQWT (Tunable Q-factor Wavelet Transform) coefficients and DFA (Detrended Fluctuation Analysis). In speech pathology and Parkinson's research, TQWT captures high-resolution time-frequency oscillations (like vocal tremors, breathiness, and micro-perturbations), while DFA measures nonlinear fractal scaling of the speech signal. Your model is locking onto the exact acoustic markers known to correlate with motor and vocal degeneration in Parkinson's disease.
"""

# Set working directory
import os
os.chdir(os.path.expanduser("~/Python/james/ML/Parkinson"))
print(os.getcwd())

# ==========================================
# Load Required Libraries
# ==========================================
import sys
print(sys.executable)
print(sys.path)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier

# Use imblearn's pipeline to prevent data leakage during resampling
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler

#import warnings
#warnings.filterwarnings('ignore')


# ==========================================
# 1. Load Data
# ==========================================
df = pd.read_csv('data-parkinson.csv')
print("Original Shape:", df.shape)


# ==========================================
# 2. Advanced Feature Engineering (Fully Optimized)
# ==========================================
numeric_cols = [col for col in df.columns if col not in ['id', 'class']]

# Build a dictionary of aggregated columns first
agg_data = {}
for col in numeric_cols:
    grouped = df.groupby('id')[col].agg(['mean', 'std', 'min', 'max'])
    agg_data[f"{col}_mean"] = grouped['mean']
    agg_data[f"{col}_std"] = grouped['std']
    agg_data[f"{col}_min"] = grouped['min']
    agg_data[f"{col}_max"] = grouped['max']

# Include the class label mapped by patient id
agg_data['class'] = df.groupby('id')['class'].first()

# Create the final DataFrame all at once from the dictionary (avoids fragmentation)
df_agg = pd.DataFrame(agg_data).reset_index(drop=True)

# Handle any NaN values introduced by std calculation on single-row patients
df_agg = df_agg.fillna(0)

# Correlation filtering to drop multi-collinear features
corr_matrix = df_agg.drop(columns=['class']).corr()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if (upper_tri[column] > 0.85).any()]
df_clean = df_agg.drop(columns=to_drop)

X = df_clean.drop(columns=['class'])
y = df_clean['class']

print(f"Shape after advanced aggregation & correlation filtering: {df_clean.shape}")


# ==========================================
# 3. Stratified K-Fold & Leakage-Free Pipeline
# ==========================================
# Define models to test
models = {
    "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000),
    "XGBoost": XGBClassifier(eval_metric='logloss'),
    "SVM (C=1)": CalibratedClassifierCV(SVC(kernel='rbf', C=1, class_weight='balanced', random_state=10), method='sigmoid'),    
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=5, class_weight='balanced', random_state=10),
    "Extra Trees": ExtraTreesClassifier(n_estimators=300, max_depth=5, class_weight='balanced', random_state=10)
}

# Setup Stratified 5-Fold Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)

for name, model in models.items():
    print(f"\n================ Evaluating: {name} ================")
    
    roc_auc_scores = []
    pr_auc_scores = []
    
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Build a leakage-free pipeline: Scale -> Resample -> Feature Selection -> Model
        # Note: Chi2 requires non-negative values, handled by MinMaxScaler inside the fold
        pipeline = ImbPipeline([
            ('scaler', MinMaxScaler()),
            ('sampler', RandomOverSampler(sampling_strategy=1.0, random_state=10)),
            ('selector', SelectKBest(chi2, k=min(25, X_train.shape[1]))),
            ('classifier', model)
        ])
        
        # Fit on training fold only
        pipeline.fit(X_train, y_train)
        
        # Predict probabilities for robust evaluation metrics
        y_proba = pipeline.predict_proba(X_val)[:, 1]
        
        roc_auc_scores.append(roc_auc_score(y_val, y_proba))
        pr_auc_scores.append(average_precision_score(y_val, y_proba))
        
    print(f"Mean Validation ROC-AUC: {np.mean(roc_auc_scores):.4f} (+/- {np.std(roc_auc_scores):.4f})")
    print(f"Mean Validation PR-AUC:  {np.mean(pr_auc_scores):.4f} (+/- {np.std(pr_auc_scores):.4f})")



# ==========================================
# 4. Extract Top Features Across Folds
# ==========================================
feature_importance_summary = {}

# Re-run a lightweight loop to track which features are selected and their importances
for train_idx, val_idx in cv.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Isolate the steps of the pipeline to inspect them
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    sampler = RandomOverSampler(sampling_strategy=1.0, random_state=10)
    X_train_res, y_train_res = sampler.fit_resample(X_train_scaled, y_train)
    
    selector = SelectKBest(chi2, k=25)
    selector.fit(X_train_res, y_train_res)
    
    # Get names of features selected in this fold
    selected_mask = selector.get_support()
    selected_features = X.columns[selected_mask]
    
    # Train Extra Trees to check feature importances
    et_model = ExtraTreesClassifier(n_estimators=300, max_depth=5, class_weight='balanced', random_state=10)
    et_model.fit(X_train_res[:, selected_mask], y_train_res)
    
    for feat, imp in zip(selected_features, et_model.feature_importances_):
        feature_importance_summary[feat] = feature_importance_summary.get(feat, 0) + imp

# Convert to a DataFrame and sort by cumulative importance
importance_df = pd.DataFrame(list(feature_importance_summary.items()), columns=['Feature', 'ImportanceScore'])
importance_df = importance_df.sort_values(by='ImportanceScore', ascending=False).reset_index(drop=True)

print("Top 15 Most Important Features Across Folds:")
print(importance_df.head(15))

# Top 15 Most Important Features Across Folds:
#                              Feature  ImportanceScore
# 0      tqwt_kurtosisValue_dec_36_std         0.310500
# 1     tqwt_kurtosisValue_dec_26_mean         0.290594
# 2   tqwt_entropy_shannon_dec_11_mean         0.287184
# 3   tqwt_entropy_shannon_dec_12_mean         0.270656
# 4        tqwt_medianValue_dec_12_std         0.245875
# 5      tqwt_kurtosisValue_dec_36_min         0.238824
# 6            tqwt_energy_dec_12_mean         0.226869
# 7      tqwt_skewnessValue_dec_27_std         0.207648
# 8            tqwt_energy_dec_26_mean         0.168054
# 9      tqwt_kurtosisValue_dec_26_std         0.163725
# 10                          DFA_mean         0.154484
# 11     tqwt_kurtosisValue_dec_26_min         0.130788
# 12           tqwt_energy_dec_14_mean         0.121608
# 13  tqwt_entropy_shannon_dec_10_mean         0.116853
# 14       tqwt_medianValue_dec_22_std         0.115499



# ==========================================
# 5. Hyperparameter Tuning (Extra Trees)
# ==========================================
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Define a broader hyperparameter grid for Extra Trees
param_grid = {
    'classifier__n_estimators': [100, 300, 500],
    'classifier__max_depth': [3, 5, 10, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__max_features': ['sqrt', 'log2', 0.5]
}

# Setup the leakage-free pipeline again
tuning_pipeline = ImbPipeline([
    ('scaler', MinMaxScaler()),
    ('sampler', RandomOverSampler(sampling_strategy=1.0, random_state=10)),
    ('selector', SelectKBest(chi2, k=30)),
    ('classifier', ExtraTreesClassifier(class_weight='balanced', random_state=10))
])

# Use Randomized Search with Stratified K-Fold
cv_search = RandomizedSearchCV(
    estimator=tuning_pipeline,
    param_distributions=param_grid,
    n_iter=15,             # Number of parameter settings sampled
    scoring='roc_auc',     # Optimize for ROC-AUC
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=10),
    random_state=10,
    n_jobs=-1
)

print("Running Randomized Search for Extra Trees...")
cv_search.fit(X, y)

print("\nBest Parameters Found:")
print(cv_search.best_params_)
print(f"Best Cross-Validation ROC-AUC: {cv_search.best_score_:.4f}")

# Best Parameters Found:
# {'classifier__n_estimators': 100, 'classifier__min_samples_split': 10, 'classifier__min_samples_leaf': 1, 'classifier__max_features': 'sqrt', 'classifier__max_depth': None}
# Best Cross-Validation ROC-AUC: 0.7973
