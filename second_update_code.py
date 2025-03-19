from collections import defaultdict
import os
import re
from imblearn.over_sampling import SMOTE
import json
import subprocess
import pandas as pd
import radon.complexity as radon_complexity
import ast
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
)
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
import spacy
import requests
import time
from requests.exceptions import RequestException

GITHUB_API_URL = "https://api.github.com/repos"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
headers = {"Authorization": f"token {GITHUB_TOKEN}"}

REPOSITORIES = [
    "facebook/react",
    "microsoft/vscode",
    "django/django",
    "spring-projects/spring-framework",
    "kubernetes/kubernetes",
    "opencv/opencv",
]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
nlp = spacy.load("en_core_web_sm")

LANGUAGE_KEYWORDS = {
    "py": ["False", "None", "True", "and", "as", "assert", "break", "class", "continue",
           "def", "del", "elif", "else", "except", "finally", "for", "from", "global",
           "if", "import", "in", "is", "lambda", "nonlocal", "not", "or", "pass",
           "raise", "return", "try", "while", "with", "yield"],
    "java": ["abstract", "assert", "boolean", "break", "byte", "case", "catch", "char",
             "class", "const", "continue", "default", "do", "double", "else", "enum",
             "extends", "final", "finally", "float", "for", "goto", "if", "implements",
             "import", "instanceof", "int", "interface", "long", "native", "new", "null",
             "package", "private", "protected", "public", "return", "short", "static",
             "strictfp", "super", "switch", "synchronized", "this", "throw", "throws",
             "transient", "try", "void", "volatile", "while"],
    "js": ["await", "break", "case", "catch", "class", "const", "continue", "debugger",
           "default", "delete", "do", "else", "enum", "export", "extends", "finally",
           "for", "function", "if", "import", "in", "instanceof", "new", "null", "return",
           "super", "switch", "this", "throw", "try", "true", "false", "undefined",
           "var", "void", "while", "with", "yield"],
    "ts": ["abstract", "any", "as", "asserts", "async", "await", "boolean", "break",
           "case", "catch", "class", "const", "continue", "declare", "default", "delete",
           "do", "else", "enum", "export", "extends", "false", "finally", "for", "from",
           "function", "if", "implements", "import", "in", "infer", "instanceof", "interface",
           "is", "keyof", "let", "module", "namespace", "never", "new", "null", "number",
           "object", "package", "private", "protected", "public", "readonly", "record",
           "return", "super", "switch", "symbol", "template", "this", "throw", "true",
           "try", "type", "undefined", "union", "unknown", "var", "void", "while", "with",
           "yield"],
    "go": ["break", "case", "chan", "const", "continue", "default", "defer", "else",
           "fallthrough", "for", "func", "go", "goto", "if", "import", "interface",
           "map", "package", "range", "return", "select", "struct", "switch", "type",
           "var"],
    "php": ["abstract", "and", "array", "as", "break", "callable", "case", "catch", "class",
            "clone", "const", "continue", "declare", "default", "die", "do", "echo", "else",
            "elseif", "empty", "enddeclare", "endfor", "endforeach", "endif", "endswitch",
            "endwhile", "eval", "exit", "extends", "final", "finally", "for", "foreach",
            "function", "global", "goto", "if", "implements", "include", "include_once",
            "instanceof", "insteadof", "interface", "isset", "list", "namespace", "new",
            "or", "print", "private", "protected", "public", "require", "require_once",
            "return", "static", "switch", "throw", "trait", "try", "unset", "use", "var",
            "while", "xor", "yield"],
    "cpp": ["alignas", "alignof", "and", "and_eq", "asm", "auto", "bitand", "bitor",
            "bool", "break", "case", "catch", "char", "char16_t", "char32_t", "class",
            "compl", "concept", "const", "const_cast", "continue", "decltype", "default",
            "delete", "do", "double", "dynamic_cast", "else", "enum", "explicit", "export",
            "extern", "false", "float", "for", "friend", "goto", "if", "inline", "int",
            "long", "mutable", "namespace", "new", "noexcept", "not", "not_eq", "nullptr",
            "operator", "or", "or_eq", "private", "protected", "public", "register",
            "reinterpret_cast", "return", "short", "signed", "sizeof", "static", "static_assert",
            "static_cast", "struct", "switch", "template", "this", "thread_local", "throw",
            "true", "try", "typedef", "typeid", "typename", "union", "unsigned", "using",
            "virtual", "void", "volatile", "wchar_t", "while", "xor", "xor_eq"]
}

LABEL_KEYWORDS = {
    "bug": [
        "bug",
        "error",
        "issue",
        "fix",
        "crash",
        "fail",
        "wrong",
        "incorrect",
        "problem",
    ],
    "feature": [
        "feature",
        "new",
        "add",
        "introduce",
        "implement",
        "provide",
        "create",
        "integrate",
    ],
    "enhancement": [
        "enhancement",
        "improve",
        "upgrade",
        "optimize",
        "better",
        "expand",
        "develop",
        "refine",
    ],
    "documentation": [
        "doc",
        "documentation",
        "comment",
        "guide",
        "help",
        "manual",
        "describe",
        "explain",
    ],
}


def fetch_data_with_retries(url, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response
        except (RequestException, ConnectionError) as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise e  # Raise exception if the maximum retries are reached


def fetch_pull_requests(repo_owner, repo_name, state="all"):
    url = f"{GITHUB_API_URL}/{repo_owner}/{repo_name}/pulls"
    params = {"state": state}
    response = fetch_data_with_retries(url)
    return response.json()


def label_issue_or_pr(issue):
    content = (issue["title"] + " " + issue["body"]).lower()

    # Split content into manageable chunks
    chunk_size = 500000  # Adjust the chunk size as needed
    chunks = [content[i : i + chunk_size] for i in range(0, len(content), chunk_size)]

    label_counts = defaultdict(int)

    for chunk in chunks:
        doc = nlp(chunk)

        for token in doc:
            for label, keywords in LABEL_KEYWORDS.items():
                if any(keyword in token.text for keyword in keywords):
                    label_counts[label] += 1

    if label_counts:
        return max(label_counts, key=label_counts.get)

    return None  # Return None if no label matches


def parse_java_classes(content):
    tree = ast.parse(content)
    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    return classes


def calculate_code_complexity(content):
    functions = radon_complexity.cc_visit(content)
    return {func.name: func.complexity for func in functions}


def fetch_github_file(repo_owner, repo_name, file_path, branch="main"):
    url = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{branch}/{file_path}"
    response = fetch_data_with_retries(url)
    return response.text


def extract_java_methods(content):
    methods = []
    tree = ast.parse(content)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            methods.append(node.name)
    return methods


def fetch_and_label_issues():
    all_issues = []

    for repo in REPOSITORIES:
        repo_owner, repo_name = repo.split("/")
        logging.info(f"Fetching issues for repository: {repo}")
        issues = fetch_pull_requests(repo_owner, repo_name)
        labeled_issues = [
            {**issue, "label": label_issue_or_pr(issue)} for issue in issues
        ]
        all_issues.extend(labeled_issues)

    return all_issues


def extract_and_store_code_metrics(all_issues):
    code_metrics = []

    for issue in all_issues:
        if issue.get("label") in ("bug", "enhancement", "feature"):
            repo_owner, repo_name = issue["repository_url"].split("/")[-2:]
            file_path = (
                issue["pull_request"]["url"].split("/")[-1].split("?")[0] + ".java"
            )

            try:
                logging.info(f"Fetching file for issue: {issue['id']}")
                file_content = fetch_github_file(repo_owner, repo_name, file_path)
                complexity = calculate_code_complexity(file_content)
                java_methods = extract_java_methods(file_content)

                code_metrics.append(
                    {
                        "issue_id": issue["id"],
                        "label": issue["label"],
                        "complexity": complexity,
                        "methods": java_methods,
                    }
                )
            except Exception as e:
                logging.error(f"Failed to fetch or process file: {e}")

    return code_metrics


def process_and_store_code_metrics():
    all_issues = fetch_and_label_issues()
    code_metrics = extract_and_store_code_metrics(all_issues)

    # Save to a JSON file
    with open("code_metrics.json", "w") as f:
        json.dump(code_metrics, f, indent=2)


def load_data():
    data = pd.read_json("code_metrics.json")
    return data


def preprocess_data(data):
    # Handle missing values
    data["complexity"] = data["complexity"].apply(lambda x: sum(x.values()) if isinstance(x, dict) else 0)

    # Transform 'methods' into feature
    data["num_methods"] = data["methods"].apply(lambda x: len(x) if isinstance(x, list) else 0)

    # Drop unnecessary columns
    data = data.drop(columns=["methods"])

    # Impute missing values
    imputer = SimpleImputer(strategy="mean")
    data[["complexity", "num_methods"]] = imputer.fit_transform(data[["complexity", "num_methods"]])

    # Convert categorical labels to numeric
    label_mapping = {"bug": 0, "enhancement": 1, "feature": 2, "documentation": 3}
    data["label"] = data["label"].map(label_mapping)

    # Drop rows with missing labels
    data = data.dropna(subset=["label"])

    X = data.drop(columns=["label"])
    y = data["label"]

    return X, y


def main():
    process_and_store_code_metrics()
    data = load_data()
    X, y = preprocess_data(data)

    # Apply feature selection
    selector = SelectKBest(f_classif, k="all")
    X = selector.fit_transform(X, y)

    # Handle imbalanced data
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )

    # Define classifiers
    rf_clf = RandomForestClassifier(random_state=42)
    svc_clf = SVC(probability=True, random_state=42)
    mlp_clf = MLPClassifier(random_state=42, max_iter=300)

    # Create a stacking classifier
    stack_clf = StackingClassifier(
        estimators=[("rf", rf_clf), ("svc", svc_clf), ("mlp", mlp_clf)],
        final_estimator=LogisticRegression(),
        cv=StratifiedKFold(n_splits=5)
    )

    # Define hyperparameters for grid search
    model_params = {
        "rf__n_estimators": [100, 200],
        "svc__C": [0.1, 1],
        "mlp__hidden_layer_sizes": [(50, 50), (100, 100)],
    }

    grid_search = GridSearchCV(stack_clf, model_params, cv=StratifiedKFold(n_splits=5))
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Make predictions
    y_pred = best_model.predict(X_test)
    y_pred_prob = best_model.predict_proba(X_test)

    # If binary classification, use only the second column for ROC AUC
    if y_pred_prob.shape[1] > 2:
        roc_auc = roc_auc_score(y_test, y_pred_prob, multi_class="ovo")
    else:
        roc_auc = roc_auc_score(y_test, y_pred_prob[:, 1])

    # Evaluate the model
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted')}")
    print(f"Recall: {recall_score(y_test, y_pred, average='weighted')}")
    print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted')}")
    print(f"ROC AUC Score: {roc_auc}")

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_mapping.keys(), yticklabels=label_mapping.keys())
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


if __name__ == "__main__":
    main()
