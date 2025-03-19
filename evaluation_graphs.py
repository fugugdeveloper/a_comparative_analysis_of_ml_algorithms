import os
from imblearn.over_sampling import SMOTE
import json
import subprocess
import pandas as pd
import radon.complexity as radon_complexity
import ast
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, auc
from sklearn.linear_model import LogisticRegression
from concurrent.futures import ProcessPoolExecutor
from sklearn.feature_selection import SelectKBest, f_classif
import logging
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from itertools import cycle
from sklearn.impute import SimpleImputer
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Directory where repositories are already cloned
CLONE_DIR = 'cloned_repos'

# Keywords to identify specific types of issues
KEYWORDS = {
    'bug': ['bug', 'fix', 'error', 'issue'],
    'code_smell': ['smell', 'cleanup', 'refactor', 'improve'],
    'security': ['security', 'vulnerability', 'exploit', 'attack']
}

def get_pylint_metrics(file_path):
    try:
        result = subprocess.run(['pylint', file_path, '-f', 'json'], capture_output=True, text=True, check=True)
        pylint_output = json.loads(result.stdout)
        metrics = {
            'num_errors': sum(1 for issue in pylint_output if issue['type'] == 'error'),
            'num_warnings': sum(1 for issue in pylint_output if issue['type'] == 'warning'),
            'num_refactors': sum(1 for issue in pylint_output if issue['type'] == 'refactor'),
            'num_convention_issues': sum(1 for issue in pylint_output if issue['type'] == 'convention')
        }
    except subprocess.CalledProcessError as e:
        logging.warning(f"Pylint failed for {file_path}: {e}")
        metrics = {
            'num_errors': 0,
            'num_warnings': 0,
            'num_refactors': 0,
            'num_convention_issues': 0
        }
    return metrics

def compute_complexity_metrics(file_path):
    try:
        with open(file_path, 'r', errors='ignore') as f:
            code = f.read()
        # Compute cyclomatic complexity using Radon
        cc_results = radon_complexity.cc_visit(code)
        avg_complexity = sum(block.complexity for block in cc_results) / len(cc_results) if cc_results else 0
        return {'cyclomatic_complexity': avg_complexity}
    except Exception as e:
        logging.error(f"Error computing complexity for {file_path}: {e}")
        return {'cyclomatic_complexity': 0}

def analyze_import_statements(file_path):
    try:
        with open(file_path, 'r', errors='ignore') as f:
            tree = ast.parse(f.read(), filename=file_path)
        imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom)]
        num_imports = len(imports)
        return {'num_imports': num_imports}
    except Exception as e:
        logging.error(f"Error analyzing imports for {file_path}: {e}")
        return {'num_imports': 0}

def extract_basic_features(file_path):
    try:
        with open(file_path, 'r', errors='ignore') as f:
            lines = f.readlines()
        num_lines = len(lines)
        num_functions = sum(1 for line in lines if 'def ' in line or 'function ' in line)
        num_comments = sum(1 for line in lines if line.strip().startswith('#') or line.strip().startswith('//'))
        num_keywords = sum(1 for line in lines if any(keyword in line for keyword in KEYWORDS['bug'] + KEYWORDS['code_smell'] + KEYWORDS['security']))
        return {
            'num_lines': num_lines,
            'num_functions': num_functions,
            'num_comments': num_comments,
            'num_keywords': num_keywords
        }
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        return {
            'num_lines': 0,
            'num_functions': 0,
            'num_comments': 0,
            'num_keywords': 0
        }

def label_issue_or_pr(issue):
    title = issue.get('title', '').lower()
    body = issue.get('body', '').lower()
    labels = set()
    for label, keywords in KEYWORDS.items():
        if any(keyword in title or keyword in body for keyword in keywords):
            labels.add(label)
    return ', '.join(labels) if labels else 'other'

def extract_features_and_labels(repo_path):
    data = []
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if file.endswith(('.py', '.js', '.java', '.go', '.rb', '.php','.ts')):  # Add more extensions as needed
                file_path = os.path.join(root, file)
                logging.info(f"Processing file: {file_path}")
                
                # Extract basic features
                features = extract_basic_features(file_path)

                # Compute complexity metrics for Python files
                if file.endswith('.py'):
                    pylint_metrics = get_pylint_metrics(file_path)
                    complexity_metrics = compute_complexity_metrics(file_path)
                    import_metrics = analyze_import_statements(file_path)
                    features.update(pylint_metrics)
                    features.update(complexity_metrics)
                    features.update(import_metrics)
                
                labels = label_issue_or_pr({'title': file_path, 'body': open(file_path, 'r', errors='ignore').read()})
                features['labels'] = labels
                data.append(features)
                logging.info(f"Extracted features for {file_path}: {features}")
    return data

# List of popular repositories to analyze (directories within cloned_repos)
REPOSITORIES = [
    'react-main'
    'vscode-main/vscode-main',
    'django-main/django-main',
    'spring-framework-main/spring-framework-main',
    'kubernetes-master/kubernetes-master'
]

def process_repository(repo):
    repo_path = os.path.join(CLONE_DIR, repo)
    logging.info(f"Analyzing repository {repo_path}...")
    repo_data = extract_features_and_labels(repo_path)
    for record in repo_data:
        record['repository'] = repo
    return repo_data

def evaluate_model(model, X_train, X_test, y_train, y_test):
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovo', average='weighted')
    
    logging.info(f"Model: {model.__class__.__name__}")
    logging.info(f"Accuracy: {accuracy}")
    logging.info(f"Precision: {precision}")
    logging.info(f"Recall: {recall}")
    logging.info(f"F1 Score: {f1}")
    logging.info(f"ROC-AUC: {roc_auc}")
    logging.info(f"Training Time: {training_time}s")
    logging.info(f"Prediction Time: {prediction_time}s")
    logging.info(f"Interpretability: {evaluate_interpretability(model)}")
    
    return {
        'model': model.__class__.__name__,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'training_time': training_time,
        'prediction_time': prediction_time,
        'interpretability': evaluate_interpretability(model),
        'y_pred': y_pred,
        'y_test': y_test,
        'probs': model.predict_proba(X_test)
    }

def evaluate_interpretability(model):
    if isinstance(model, RandomForestClassifier):
        return "Medium (feature importance available)"
    elif isinstance(model, SVC):
        return "Low (complex decision boundaries)"
    elif isinstance(model, MLPClassifier):
        return "Low (black-box model)"
    else:
        return "Unknown"

def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'confusion_matrix_{model_name}.png')
    plt.close()

def plot_roc_curve(y_test, probs, model_name):
    fpr = {}
    tpr = {}
    roc_auc = {}
    n_classes = len(set(y_test))
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test, probs[:, i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic for {model_name}')
    plt.legend(loc='lower right')
    plt.savefig(f'roc_curve_{model_name}.png')
    plt.close()

def plot_precision_recall_curve(y_test, probs, model_name):
    precision = {}
    recall = {}
    n_classes = len(set(y_test))
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test, probs[:, i], pos_label=i)

    plt.figure()
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label='PR curve of class {0}'
                 ''.format(i))

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for {model_name}')
    plt.legend(loc='lower left')
    plt.savefig(f'precision_recall_curve_{model_name}.png')
    plt.close()

def plot_metrics_comparison(results):
    df = pd.DataFrame(results)
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'training_time', 'prediction_time']
    df_melted = df.melt(id_vars=['model'], value_vars=metrics)
    
    plt.figure(figsize=(14, 7))
    sns.barplot(data=df_melted, x='variable', y='value', hue='model')
    plt.title('Model Performance Metrics Comparison')
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.xticks(rotation=45)
    plt.legend(title='Model')
    plt.tight_layout()
    plt.savefig('metrics_comparison.png')
    plt.close()

def plot_learning_curve(estimator, X, y, model_name):
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure()
    plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')
    plt.title(f'Learning Curve for {model_name}')
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.savefig(f'learning_curve_{model_name}.png')
    plt.close()



def main():
    # Use parallel processing for efficiency
    with ProcessPoolExecutor() as executor:
        all_data = []
        results = executor.map(process_repository, REPOSITORIES)
        for repo_data in results:
            all_data.extend(repo_data)
    
    # Check if any data was collected
    if not all_data:
        logging.warning("No data collected from repositories.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    logging.info(df.head())  # Debugging: print the first few rows to ensure labels are present
    df.to_csv('features_and_labels.csv', index=False)
    logging.info("Data saved to features_and_labels.csv")
    
    # Prepare features and labels
    if 'labels' not in df.columns:
        logging.error("Error: 'labels' column not found in DataFrame.")
        return
    
    X = df.drop(['labels', 'repository'], axis=1)
    y = df['labels'].astype('category').cat.codes
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')  # You can change the strategy to 'median', 'most_frequent', etc.
    X_imputed = imputer.fit_transform(X)
    
    # Feature selection
    X_new = SelectKBest(f_classif, k='all').fit_transform(X_imputed, y)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)
    
    # Check class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    class_counts = dict(zip(unique, counts))
    logging.info(f"Class distribution in y_train: {class_counts}")
    
    # Adjust k_neighbors for SMOTE
    min_class_size = min(class_counts.values())
    k_neighbors = min(5, min_class_size - 1)  # Ensure k_neighbors is smaller than the smallest class size
    
    # Apply SMOTE to the training data
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # Define models and hyperparameters
    models = {
        'RandomForest': RandomForestClassifier(),
        'SVM': SVC(probability=True),
        'NeuralNetwork': MLPClassifier(max_iter=500)
    }
    
    param_grids = {
        'RandomForest': {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5]},
        'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']},
        'NeuralNetwork': {'hidden_layer_sizes': [(100,), (100, 50)], 'activation': ['relu', 'tanh'], 'solver': ['adam', 'sgd']}
    }
    
    results = []
    for name, model in models.items():
        logging.info(f"Training {name}...")
        grid_search = GridSearchCV(model, param_grids[name], cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(X_train_res, y_train_res)
        best_model = grid_search.best_estimator_
        
        # Evaluate model
        result = evaluate_model(best_model, X_train_res, X_test, y_train_res, y_test)
        results.append(result)
        
        # Plot Confusion Matrix
        cm = confusion_matrix(y_test, result['y_pred'])
        plot_confusion_matrix(cm, name)
        
        # Plot ROC Curve
        plot_roc_curve(y_test, result['probs'], name)
        
        # Plot Precision-Recall Curve
        plot_precision_recall_curve(y_test, result['probs'], name)
        
        # Plot Learning Curve
        plot_learning_curve(best_model, X, y, name)
    
    # Ensemble Methods - Stacking
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=100)),
        ('svm', SVC(kernel='linear', probability=True)),
        ('mlp', MLPClassifier(max_iter=500))
    ]
    
    stacking_model = StackingClassifier(
        estimators=estimators, final_estimator=LogisticRegression()
    )
    
    logging.info("Training Stacking Model...")
    result = evaluate_model(stacking_model, X_train_res, X_test, y_train_res, y_test)
    results.append(result)
    
    # Plot Confusion Matrix for Stacking Model
    cm = confusion_matrix(y_test, result['y_pred'])
    plot_confusion_matrix(cm, 'Stacking')
    
    # Plot ROC Curve for Stacking Model
    plot_roc_curve(y_test, result['probs'], 'Stacking')
    
    # Plot Precision-Recall Curve for Stacking Model
    plot_precision_recall_curve(y_test, result['probs'], 'Stacking')
    
    # Plot Learning Curve for Stacking Model
    plot_learning_curve(stacking_model, X, y, 'Stacking')
    
    # Save results to a DataFrame and output
    results_df = pd.DataFrame(results)
    logging.info(results_df)
    results_df.to_csv('model_performance_comparison.csv', index=False)
    logging.info("Model performance comparison saved to model_performance_comparison.csv")
    
    # Plot comparison of metrics
    plot_metrics_comparison(results)

if __name__ == "__main__":
    main()


