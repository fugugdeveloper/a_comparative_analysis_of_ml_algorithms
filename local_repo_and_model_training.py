import os
import json
import subprocess
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

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
    except subprocess.CalledProcessError:
        metrics = {
            'num_errors': 0,
            'num_warnings': 0,
            'num_refactors': 0,
            'num_convention_issues': 0
        }
    return metrics

def extract_basic_features(file_path):
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
                print(f"Processing file: {file_path}")  # Debugging: print each file being processed
                features = extract_basic_features(file_path)
                if file.endswith('.py'):
                    pylint_metrics = get_pylint_metrics(file_path)
                    features.update(pylint_metrics)
                labels = label_issue_or_pr({'title': file_path, 'body': open(file_path, 'r', errors='ignore').read()})
                features['labels'] = labels
                data.append(features)
                print(f"Extracted features for {file_path}: {features}")  # Debugging: print extracted features
    return data

# List of popular repositories to analyze (directories within cloned_repos)
REPOSITORIES = [
    'react-main'
    # 'vscode-main/vscode-main',
    # 'django-main/django-main',
    # 'spring-framework-main/spring-framework-main',
    # 'react-main',
    # 'kubernetes-master/kubernetes-master'
]

def main():
    all_data = []
    for repo in REPOSITORIES:
        repo_path = os.path.join(CLONE_DIR, repo)
        print(f"Analyzing repository {repo_path}...")
        repo_data = extract_features_and_labels(repo_path)
        for record in repo_data:
            record['repository'] = repo
        all_data.extend(repo_data)
    
    # Check if any data was collected
    if not all_data:
        print("No data collected from repositories.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    print(df.head())  # Debugging: print the first few rows to ensure labels are present
    df.to_csv('features_and_labels.csv', index=False)
    print("Data saved to features_and_labels.csv")
    
    # Prepare features and labels
    if 'labels' not in df.columns:
        print("Error: 'labels' column not found in DataFrame.")
        return
    
    X = df.drop('labels', axis=1)
    y = df['labels'].astype('category').cat.codes
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define models and hyperparameters
    models = {
        'RandomForest': RandomForestClassifier(),
        'SVM': SVC(),
        'NeuralNetwork': MLPClassifier(max_iter=500)
    }
    
    param_grids = {
        'RandomForest': {'n_estimators': [100, 200], 'max_depth': [10, 20, None]},
        'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
        'NeuralNetwork': {'hidden_layer_sizes': [(100,), (100, 50)], 'activation': ['relu', 'tanh']}
    }
    
    for name, model in models.items():
        print(f"Training {name}...")
        grid_search = GridSearchCV(model, param_grids[name], cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        # Cross-validation scores
        scores = cross_val_score(best_model, X, y, cv=5)
        print(f"Cross-validation scores for {name}: {scores}")
        
        # Evaluate on test set
        y_pred = best_model.predict(X_test)
        print(f"Performance of {name} on test set:\n")
        print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
