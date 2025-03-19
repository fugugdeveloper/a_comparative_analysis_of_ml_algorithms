import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import ijson
import logging
from pandas import json_normalize
import json
import dask.dataframe as dd

logging.basicConfig(level=logging.DEBUG)
# Load the dataset
# Assume dataset is in CSV format with appropriate columns
# data = pd.read_csv('cross_language_code_quality_dataset.csv')
file_path = 'cloned_repos/ghtorrent-2019-01-07.csv/ghtorrent-2019-01-07.csv'




# Read the CSV file using Dask
df = dd.read_csv(file_path)

# Perform operations on the Dask DataFrame
#result = df.groupby('some_column').mean()  # Example operation

# Compute the result (Dask operations are lazy)
result_computed = df.compute()
print("Computed_Result: ", result_computed)

# Optionally, save the result

# Option 2: Out-of-Core Computation with dask (Alternative to ijson approach)
# ddf = dd.read_json(file_path, lines=True)
# df = ddf.compute()

# Feature and target selection
X = df.drop(columns=['Label'])
y = df['Label']
# Feature and target selection

# Handling missing values
X.fillna(X.median(), inplace=True)  # Numerical features
y.fillna(y.mode()[0], inplace=True)  # Categorical features

# Feature selection
selector = SelectKBest(score_func=f_classif, k=20)  # Adjust k based on the number of features you want to select
X_selected = selector.fit_transform(X, y)

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Addressing data imbalance
over_sampler = SMOTE(sampling_strategy='minority', random_state=42)
under_sampler = RandomUnderSampler(sampling_strategy='majority', random_state=42)
steps = [('over', over_sampler), ('under', under_sampler)]
pipeline = Pipeline(steps=steps)

X_train_balanced, y_train_balanced = pipeline.fit_resample(X_train, y_train)

# Model training
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42, probability=True),
    'NeuralNetwork': MLPClassifier(random_state=42, max_iter=300)
}

# Hyperparameter tuning (example for RandomForest, extend for other models)
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid_rf = GridSearchCV(estimator=models['RandomForest'], param_grid=param_grid_rf, cv=5, scoring='accuracy')
grid_rf.fit(X_train_balanced, y_train_balanced)
best_rf = grid_rf.best_estimator_

# Model Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred)
    return accuracy, precision, recall, report

# Evaluate Random Forest
accuracy_rf, precision_rf, recall_rf, report_rf = evaluate_model(best_rf, X_test, y_test)

# Train and evaluate SVM
grid_svm = GridSearchCV(estimator=models['SVM'], param_grid={'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}, cv=5, scoring='accuracy')
grid_svm.fit(X_train_balanced, y_train_balanced)
best_svm = grid_svm.best_estimator_

accuracy_svm, precision_svm, recall_svm, report_svm = evaluate_model(best_svm, X_test, y_test)

# Train and evaluate Neural Network
grid_nn = GridSearchCV(estimator=models['NeuralNetwork'], param_grid={'hidden_layer_sizes': [(50,50), (100,)], 'alpha': [0.0001, 0.001]}, cv=5, scoring='accuracy')
grid_nn.fit(X_train_balanced, y_train_balanced)
best_nn = grid_nn.best_estimator_

accuracy_nn, precision_nn, recall_nn, report_nn = evaluate_model(best_nn, X_test, y_test)

# Print evaluation results
print("Random Forest Evaluation:")
print(f"Accuracy: {accuracy_rf}, Precision: {precision_rf}, Recall: {recall_rf}")
print(report_rf)

print("SVM Evaluation:")
print(f"Accuracy: {accuracy_svm}, Precision: {precision_svm}, Recall: {recall_svm}")
print(report_svm)

print("Neural Network Evaluation:")
print(f"Accuracy: {accuracy_nn}, Precision: {precision_nn}, Recall: {recall_nn}")
print(report_nn)
