import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
             'marital-status', 'occupation', 'relationship', 'race', 'sex',
             'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
df = pd.read_csv('adult.data', header=None, names=col_names)

# Clean columns by stripping extra whitespace
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

#Feature selection (before binning education)
feature_cols = ['age', 'capital-gain', 'capital-loss', 'hours-per-week', 'sex', 'race', 'education-num']
X = pd.get_dummies(df[feature_cols], drop_first=True)
y = df['income'].apply(lambda x: 0 if x == "<=50K" else 1)

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

# Random forest classifier (baseline)
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
print("Baseline Accuracy:", rf.score(x_test, y_test))

# Tune max_depth (Baseline)
accuracy_train, accuracy_test = [], []
for i in range(1, 26):
    rf1 = RandomForestClassifier(max_depth=i, n_estimators=50, n_jobs=-1)
    rf1.fit(x_train, y_train)
    accuracy_train.append(rf1.score(x_train, y_train))
    accuracy_test.append(rf1.score(x_test, y_test))

print("Best accuracy:", np.max(accuracy_test), "at depth:", np.argmax(accuracy_test) + 1)

# Plot accuracy
plt.plot(range(1, 26), accuracy_test, label="Test Accuracy")
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.title("Max Depth Tuning")
plt.legend()
plt.show()

# Save best RF model and feature importances
best_rf = RandomForestClassifier(max_depth=10)
best_rf.fit(x_train, y_train)
feat_importance = pd.DataFrame({'Features': x_train.columns, 'Importance': best_rf.feature_importances_})
feat_importance = feat_importance.sort_values(by='Importance', ascending=False)
print("\nTop 5 Important Features:\n", feat_importance.head())

#-------------------------------
#Feature Engineering: Education Binning
#-------------------------------
education_bins = {
    'Preschool': 'High school and less',
    '1st-4th': 'High school and less',
    '5th-6th': 'High school and less',
    '7th-8th': 'High school and less',
    '9th': 'High school and less',
    '10th': 'High school and less',
    '11th': 'High school and less',
    '12th': 'High school and less',
    'HS-grad': 'High school and less',
    'Some-college': 'College to Bachelors',
    'Assoc-acdm': 'College to Bachelors',
    'Assoc-voc': 'College to Bachelors',
    'Bachelors': 'College to Bachelors',
    'Masters': 'Masters and more',
    'Doctorate': 'Masters and more',
    'Prof-school': 'Masters and more'
}

df['education_bin'] = df['education'].replace(education_bins)

# Feature selection (after binning)
feature_cols = ['age', 'capital-gain', 'capital-loss', 'hours-per-week', 'sex', 'race', 'education_bin']
X = pd.get_dummies(df[feature_cols], drop_first=True)
y = df['income'].apply(lambda x: 0 if x == "<=50K" else 1)

# Train/test split (with new features)
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# Tune max_depth again
accuracy_train, accuracy_test = [], []
for i in range(1, 26):
    rf1 = RandomForestClassifier(max_depth=i, n_estimators=50, n_jobs=-1)
    rf1.fit(x_train, y_train)
    accuracy_train.append(rf1.score(x_train, y_train))
    accuracy_test.append(rf1.score(x_test, y_test))

print("Best accuracy with new features:", np.max(accuracy_test), "at depth:", np.argmax(accuracy_test) + 1)
