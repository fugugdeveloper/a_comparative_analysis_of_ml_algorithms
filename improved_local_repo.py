import os
import json
import subprocess
import pandas as pd
import radon.complexity as radon_complexity
import ast
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from concurrent.futures import ProcessPoolExecutor
from sklearn.feature_selection import SelectKBest, f_classif
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Directory where repositories are located 
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
        num_comments = sum(1 for line in lines if line.strip().startswith('#') or line.strip().startswith('//') or line.strip().startswith('/*'))
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
            if file.endswith(('.py', '.js', '.java', '.go', '.rb', '.php')):  # Add more extensions as needed
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
    'vscode-main/vscode-main',
    'django-main/django-main',
    'spring-framework-main/spring-framework-main',
    'react-main',
    'kubernetes-master/kubernetes-master',
    'laravel-11.x/laravel-11.x',
    'gin-master/gin-master'
]

def process_repository(repo):
    repo_path = os.path.join(CLONE_DIR, repo)
    logging.info(f"Analyzing repository {repo_path}...")
    repo_data = extract_features_and_labels(repo_path)
    for record in repo_data:
        record['repository'] = repo
    return repo_data

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
    
    # Feature selection
    X_new = SelectKBest(f_classif, k='all').fit_transform(X, y)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)
    
    # Define models and hyperparameters
    models = {
        'RandomForest': RandomForestClassifier(),
        'SVM': SVC(),
        'NeuralNetwork': MLPClassifier(max_iter=500)
    }
    
    param_grids = {
        'RandomForest': {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5]},
        'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']},
        'NeuralNetwork': {'hidden_layer_sizes': [(100,), (100, 50)], 'activation': ['relu', 'tanh'], 'solver': ['adam', 'sgd']}
    }
    
    for name, model in models.items():
        logging.info(f"Training {name}...")
        grid_search = GridSearchCV(model, param_grids[name], cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        # Cross-validation scores
        scores = cross_val_score(best_model, X_new, y, cv=5)
        logging.info(f"Cross-validation scores for {name}: {scores}")
        
        # Evaluate on test set
        y_pred = best_model.predict(X_test)
        logging.info(f"Performance of {name} on test set:\n{classification_report(y_test, y_pred)}")

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
    stacking_model.fit(X_train, y_train)
    y_pred_stack = stacking_model.predict(X_test)
    logging.info(f"Performance of Stacking Model on test set:\n{classification_report(y_test, y_pred_stack)}")

if __name__ == "__main__":
    main()
