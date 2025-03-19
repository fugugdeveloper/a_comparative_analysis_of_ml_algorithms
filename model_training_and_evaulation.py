import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

def main():
    all_data = []
    for repo_url in REPOSITORIES:
        print(f"Cloning repository {repo_url}...")
        repo_path = clone_repo(repo_url, CLONE_DIR)
        print(f"Analyzing repository {repo_url}...")
        repo_data = extract_features_and_labels(repo_path)
        all_data.extend(repo_data)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    df.to_csv('features_and_labels.csv', index=False)
    print("Data saved to features_and_labels.csv")
    
    # Prepare features and labels
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
