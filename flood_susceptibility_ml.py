
# Flood Susceptibility Modeling using Morphometric Parameters and ML
# Author: Desmond R. Eteh et al.
# GitHub-ready Python Script

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Simulate morphometric dataset (replace with real data)
np.random.seed(42)
n_samples = 1000

# Morphometric features
X = pd.DataFrame({
    'Dd': np.random.uniform(1.5, 4.0, n_samples),
    'Rh': np.random.uniform(5, 25, n_samples),
    'If': np.random.uniform(10, 50, n_samples),
    'Rbm': np.random.uniform(1.5, 4.5, n_samples),
    'Fs': np.random.uniform(3, 12, n_samples),
    'Ff': np.random.uniform(0.5, 1.5, n_samples)
})

# Binary target: 1 = flood-prone, 0 = non-flood-prone
y = (X['Dd'] + X['If'] + X['Rh'] > 55).astype(int)

# Step 2: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Step 3: Train Models

# Random Forest
rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
rf.fit(X_train, y_train)

# SVM
svm = SVC(kernel='rbf', C=1, gamma=0.1, probability=True)
svm.fit(X_train, y_train)

# XGBoost
xgb = XGBClassifier(learning_rate=0.1, max_depth=6, subsample=1.0, use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)

# Step 4: Evaluate Models
def evaluate(model, name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    print(f"\n{name} Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall: {recall_score(y_test, y_pred):.3f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.3f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")
    return y_pred, y_prob

evaluate(rf, "Random Forest")
evaluate(svm, "SVM")
evaluate(xgb, "XGBoost")

# Step 5: Confusion Matrix
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
for ax, model, title in zip(axs, [rf, svm, xgb], ["Random Forest", "SVM", "XGBoost"]):
    cm = confusion_matrix(y_test, model.predict(X_test))
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig("confusion_matrices.png")

# Step 6: Feature Importance (XGBoost)
importances = xgb.feature_importances_
plt.figure(figsize=(8, 5))
plt.barh(X.columns, importances)
plt.title("XGBoost Feature Importance")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("xgboost_feature_importance.png")

# Step 7: SHAP Analysis
explainer = shap.Explainer(xgb)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png")

print("All outputs generated and saved successfully.")
