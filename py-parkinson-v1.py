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

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

#import warnings
#warnings.filterwarnings('ignore')


# ==========================================
# Load Data
# ==========================================
df = pd.read_csv('data-parkinson.csv')

pd.set_option('display.max_columns', 10)
print(df.head(5))

print("Shape:", df.shape)
print("\nInfo:")
df.info()

print("\nDescription:")
print(df.describe().T)

print("\nMissing values sum:", df.isnull().sum().sum())


# ==========================================
# Feature Engineering & Correlation Filtering
# ==========================================
# Create an aggregation dictionary: features get the mean, class gets the first value
agg_dict = {col: 'mean' for col in df.columns if col not in ['id', 'class']}
agg_dict['class'] = 'first'  # Keeps the classification label intact

# Safely group by 'id' without corrupting or losing the 'class' column
df1 = df.groupby('id').agg(agg_dict).reset_index().drop(columns=['id'])

# Calculate the correlation matrix
corr_matrix = df1.drop(columns=['class']).corr()

# Select the upper triangle of the correlation matrix
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find features with a correlation greater than 0.7
to_drop = [column for column in upper_tri.columns if (upper_tri[column] > 0.7).any()]

# Drop the highly correlated columns from your original DataFrame
df1 = df1.drop(columns=to_drop)

print("Shape after correlation drop:", df1.shape)
print("Unique classes:", df1['class'].unique())


# ==========================================
# Feature Selection (SelectKBest with Chi-Square)
# ==========================================
X = df1.drop('class', axis=1)
y = df1['class']

# Scale the features to prevent negative values (required for Chi-Square)
X_norm = MinMaxScaler().fit_transform(X)

# Apply SelectKBest with Chi-Square to pick the top 30 features
selector = SelectKBest(chi2, k=30)
selector.fit(X_norm, y)

# Filter the columns based on the selector's support mask
X_filtered_columns = selector.get_support()
X_filtered = X.loc[:, X_filtered_columns]

# Append the 'class' column back to the final DataFrame
X_filtered['class'] = y
df2 = X_filtered

print("Filtered shape:", df2.shape)


# ==========================================
# Visualization & Resampling
# ==========================================
vc = df2['class'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(vc.values, labels=vc.index, autopct='%1.1f%%')
plt.show()

print("Value counts:\n", df2['class'].value_counts())

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.4,
                                                    random_state=10)

ros = RandomOverSampler(sampling_strategy=1.0, random_state=10)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

print("Resampled shapes:", X_resampled.shape, y_resampled.value_counts())


# ==========================================
# Model Training & Evaluation
# ==========================================
models = [
    LogisticRegression(class_weight='balanced'), 
    XGBClassifier(), 
    SVC(kernel='rbf', C=1, gamma='scale', class_weight='balanced', probability=True, random_state=10),
    SVC(kernel='rbf', C=0.1, gamma='scale', class_weight='balanced', probability=True, random_state=10),
    SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced', probability=True, random_state=10),
    RandomForestClassifier(n_estimators=500, max_depth=5, min_samples_split=5, min_samples_leaf=2, max_features='sqrt', class_weight='balanced', random_state=10),
    ExtraTreesClassifier(n_estimators=500, max_depth=5, min_samples_split=5, min_samples_leaf=2, max_features='sqrt', class_weight='balanced', random_state=10)
]

for model in models:
    model.fit(X_resampled, y_resampled)
    
    print(f"{model.__class__.__name__} :")
    
    train_preds = model.predict(X_resampled)
    train_auc = roc_auc_score(y_resampled, train_preds)
    print(f"Training ROC AUC: {train_auc:.4f}")
    
    val_preds = model.predict(X_test)
    val_auc = roc_auc_score(y_test, val_preds)
    print(f"Validation ROC AUC: {val_auc:.4f}\n")

