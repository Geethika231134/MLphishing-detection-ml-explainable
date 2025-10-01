# MLphishing-detection-ml-explainable
# ML project for binary classification of phishing URLs using Random Forest with feature selection (SelectKBest), hyperparameter tuning (GridSearchCV), and advanced explainability methods (LIME &amp; SHAP) in Python. Includes scikit-learn modeling pipeline and step-by-step interpretability for security and transparency
# -- 1. Imports --
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from lime.lime_tabular import LimeTabularExplainer
import shap
import matplotlib.pyplot as plt

# -- 2. Data Preprocessing --
data = pd.read_csv("webpage_phishing.csv")
X = data.drop(['url', 'status'], axis=1)
y = data['status'].map({'phishing': 1, 'legitimate': 0})
X.fillna(0, inplace=True)

# -- 3. Feature Selection (classical, top 30 features) --
selector = SelectKBest(score_func=f_classif, k=30)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
X_selected_df = pd.DataFrame(X_selected, columns=selected_features)

# -- 4. Train-Test Split --
X_train, X_test, y_train, y_test = train_test_split(
    X_selected_df, y, test_size=0.2, random_state=42, stratify=y
)

# -- 5. Hyperparameter Tuning and Model Training --
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}
grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=3, scoring='accuracy', n_jobs=-1
)
grid.fit(X_train, y_train)
model = grid.best_estimator_

# -- 6. Evaluation --
y_pred = model.predict(X_test)
print("Random Forest - Accuracy:", accuracy_score(y_test, y_pred))
print("Random Forest - Classification Report:\n", classification_report(y_test, y_pred))
print("Random Forest - AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

# -- 7. LIME Explainability --
explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=selected_features,
    class_names=['legitimate', 'phishing'],
    discretize_continuous=True
)
idx = 0  # Index of test instance to explain
exp = explainer.explain_instance(
    X_test.iloc[idx].values, 
    model.predict_proba, 
    num_features=10
)
print("LIME explanation for test instance {}: \n".format(idx), exp.as_list())
# Optional: exp.show_in_notebook(), or exp.save_to_file('lime_explanation.html')

# -- 8. SHAP Explainability --
shap.initjs()
shap_explainer = shap.TreeExplainer(model)
shap_values = shap_explainer.shap_values(X_test)

print("SHAP values shape:", np.array(shap_values).shape)

# For binary classification: SHAP returns a list with 2 arrays (class 0 and class 1)
# Show feature importance plot for class 1 ("phishing")
if isinstance(shap_values, list):
    shap.summary_plot(shap_values[1], X_test, feature_names=selected_features)
else:
    shap.summary_plot(shap_values[:, :, 1], X_test, feature_names=selected_features)


