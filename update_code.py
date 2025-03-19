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
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV, learning_curve
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
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler

GITHUB_API_URL = "https://api.github.com/repos"
GITHUB_TOKEN = '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'
headers = {"Authorization": f"token {GITHUB_TOKEN}"}

REPOSITORIES = [
    "facebook/react",
    "microsoft/vscode",
    "django/django",
    "spring-projects/spring-framework",
    "kubernetes/kubernetes",
    'opencv/opencv'
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
                if token.lemma_ in keywords:
                    label_counts[label] += 1

        for sent in doc.sents:
            for label, keywords in LABEL_KEYWORDS.items():
                if any(keyword in sent.text.lower() for keyword in keywords):
                    label_counts[label] += 1

    if label_counts:
        most_common_label = max(label_counts, key=label_counts.get)
        if label_counts[most_common_label] > 0:
            return most_common_label

    return "other"

def extract_basic_features(file_content, language):
    loc = len(file_content.splitlines())
    comments = 0
    functions = 0
    language_keywords_count = 0

    if language == "python":
        try:
            tree = ast.parse(file_content)
            functions = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
            comments = len([node for node in ast.walk(tree) if isinstance(node, ast.Expr) and isinstance(node.value, ast.Str)])
        except:
            functions = 0
            comments = 0

    elif language in ["java", "js", "ts", "go", "php", "cpp"]:
        comments = len(re.findall(r'\/\/.*?$|\/\*[\s\S]*?\*\/', file_content, re.MULTILINE))
        functions = len(re.findall(r'\bfunction\b|\bdef\b|\bclass\b|\bmethod\b', file_content))

    keywords = LANGUAGE_KEYWORDS.get(language, [])
    language_keywords_count = sum(1 for word in file_content.split() if word in keywords)

    return {
        "loc": loc,
        "comments": comments,
        "functions": functions,
        "language_keywords_count": language_keywords_count,
    }

def process_pull_request(repo_owner, repo_name, pull_request):
    pr_number = pull_request["number"]
    files_url = pull_request["url"] + "/files"
    response = requests.get(files_url)
    files = response.json()
    logging.info(f"Data for repository {repo_name}:\n{repo_owner} files url: {files_url}")

    pr_data = []

    for file in files:
        filename = file["filename"]
        file_ext = filename.split(".")[-1]
        language = file_ext.lower()

        if language not in LANGUAGE_KEYWORDS:
            continue

        patch_url = file.get("patch_url") or file.get("raw_url")
        if patch_url:
            patch_response = requests.get(patch_url)
            file_content = patch_response.text

            features = extract_basic_features(file_content, language)
            label = label_issue_or_pr(pull_request)
            features.update({"label": label})
            pr_data.append(features)
            logging.info(f"pull request data {pr_data}")


    return pr_data

def plot_learning_curve(estimator, X_train, y_train):
    train_sizes, train_scores, test_scores = learning_curve(estimator, X_train, y_train, cv=5)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title("Learning Curve")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def plot_roc_curve(y_test, y_pred_prob):
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    return plt

def plot_precision_recall_curve(y_test, y_pred_prob):
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    pr_auc = auc(recall, precision)

    plt.figure()
    plt.plot(recall, precision, color="blue", lw=2, label=f"PR curve (area = {pr_auc:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    return plt

def main():
    for repo in REPOSITORIES:
        repo_owner, repo_name = repo.split('/')
        
        pull_requests = fetch_pull_requests(repo_owner, repo_name, state="closed")

        all_data = []
        for pr in pull_requests:
            pr_data = process_pull_request(repo_owner, repo_name, pr)
            all_data.extend(pr_data)

        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        output_filename = f"features_and_labels_{repo_name}before_evaluation.csv"
        df.to_csv(output_filename, index=False)
        logging.info(f"Data saved to {output_filename} before evaluation")
        # Debugging: print the first few rows to ensure labels are present
        logging.info(f"Data for repository {repo_name}:\n{df.head()}")

        # Prepare features and labels
        if "label" not in df.columns:
            logging.error(f"Error: 'label' column not found in DataFrame for {repo_name}.")
            continue

        X = df.drop(["label"], axis=1)
        y = df["label"].astype("category").cat.codes

        # Impute missing values
        imputer = SimpleImputer(strategy="mean")
        X_imputed = imputer.fit_transform(X)

        # Feature selection
        selector = SelectKBest(f_classif, k="all")
        X = selector.fit_transform(X, y)
        # smote = SMOTE(random_state=42)
        ros = RandomOverSampler(random_state=42)  # Using RandomOverSampler as an alternative
        X_resampled, y_resampled = ros.fit_resample(X, y)
        # X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2,
                                                            random_state=42, stratify=y_resampled)
        


        # Model training and evaluation
        rf_clf = RandomForestClassifier(random_state=42)
        svm_clf = SVC(probability=True, random_state=42)
        mlp_clf = MLPClassifier(random_state=42, max_iter=1000, solver='adam')

        stack_clf = StackingClassifier(
            estimators=[("rf", rf_clf), ("svm", svm_clf), ("mlp", mlp_clf)],
            final_estimator=LogisticRegression(),
        )

        model_params = {
            "rf__n_estimators": [100, 200],
            "rf__max_depth": [None, 10],
            "svm__C": [0.1, 1, 10],
            "mlp__hidden_layer_sizes": [(50, 50), (100, 100)],
        }
        
        grid_search = GridSearchCV(stack_clf, model_params, cv=StratifiedKFold(n_splits=3))
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_pred_prob = best_model.predict_proba(X_test)

        # If binary classification, use only the second column for ROC AUC
        # if y_pred_prob.shape[1] > 2:
        #     roc_auc = roc_auc_score(y_test, y_pred_prob, multi_class="ovo")
        # else:
        #     roc_auc = roc_auc_score(y_test, y_pred_prob[:, 1])
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")
        roc_auc = roc_auc_score(y_test, y_pred_prob, multi_class="ovr")

        logging.info(f"Best model for {repo_name}: {grid_search.best_params_}")
        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info(f"Precision: {precision:.4f}")
        logging.info(f"Recall: {recall:.4f}")
        logging.info(f"F1 Score: {f1:.4f}")
        logging.info(f"ROC AUC: {roc_auc:.4f}")

        plot_learning_curve(best_model, X_train, y_train)
        plt.savefig(f"learning_curve_{repo_name}.png")

        plot_roc_curve(y_test, y_pred_prob)
        plt.savefig(f"roc_curve_{repo_name}.png")

        plot_precision_recall_curve(y_test, y_pred_prob)
        plt.savefig(f"precision_recall_curve_{repo_name}.png")

        output_filename = f"features_and_labels_{repo_name}after_evaluation.csv"
        df.to_csv(output_filename, index=False)
        logging.info(f"Data saved to {output_filename} after evaluation")
        
if __name__ == "__main__":
    main()
