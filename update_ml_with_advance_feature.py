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
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
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

CLONE_DIR = "cloned_repos"
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
nlp = spacy.load("en_core_web_sm")

# Define keywords for Python, Java, JavaScript, Go, PHP, and C++
LANGUAGE_KEYWORDS = {
    "py": [
        "False", "None", "True", "and", "as", "assert", "break", "class", "continue",
        "def", "del", "elif", "else", "except", "finally", "for", "from", "global",
        "if", "import", "in", "is", "lambda", "nonlocal", "not", "or", "pass",
        "raise", "return", "try", "while", "with", "yield"
    ],
    "java": [
        "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char",
        "class", "const", "continue", "default", "do", "double", "else", "enum",
        "extends", "final", "finally", "float", "for", "goto", "if", "implements",
        "import", "instanceof", "int", "interface", "long", "native", "new", "null",
        "package", "private", "protected", "public", "return", "short", "static",
        "strictfp", "super", "switch", "synchronized", "this", "throw", "throws",
        "transient", "try", "void", "volatile", "while"
    ],
    "js": [
        "await", "break", "case", "catch", "class", "const", "continue", "debugger",
        "default", "delete", "do", "else", "enum", "export", "extends", "finally",
        "for", "function", "if", "import", "in", "instanceof", "new", "null", "return",
        "super", "switch", "this", "throw", "try", "true", "false", "undefined",
        "var", "void", "while", "with", "yield"
    ],
    "ts": [
        "abstract", "any", "as", "asserts", "async", "await", "boolean", "break",
        "case", "catch", "class", "const", "continue", "declare", "default", "delete",
        "do", "else", "enum", "export", "extends", "false", "finally", "for", "from",
        "function", "if", "implements", "import", "in", "infer", "instanceof", "interface",
        "is", "keyof", "let", "module", "namespace", "never", "new", "null", "number",
        "object", "package", "private", "protected", "public", "readonly", "record",
        "return", "super", "switch", "symbol", "template", "this", "throw", "true",
        "try", "type", "undefined", "union", "unknown", "var", "void", "while", "with",
        "yield"
    ],
    "go": [
        "break", "case", "chan", "const", "continue", "default", "defer", "else",
        "fallthrough", "for", "func", "go", "goto", "if", "import", "interface",
        "map", "package", "range", "return", "select", "struct", "switch", "type",
        "var"
    ],
    "php": [
        "abstract", "and", "array", "as", "break", "callable", "case", "catch", "class",
        "clone", "const", "continue", "declare", "default", "die", "do", "echo", "else",
        "elseif", "empty", "enddeclare", "endfor", "endforeach", "endif", "endswitch",
        "endwhile", "eval", "exit", "extends", "final", "finally", "for", "foreach",
        "function", "global", "goto", "if", "implements", "include", "include_once",
        "instanceof", "insteadof", "interface", "isset", "list", "namespace", "new",
        "or", "print", "private", "protected", "public", "require", "require_once",
        "return", "static", "switch", "throw", "trait", "try", "unset", "use", "var",
        "while", "xor", "yield"
    ],
    "cpp": [
        "alignas", "alignof", "and", "and_eq", "asm", "auto", "bitand", "bitor",
        "bool", "break", "case", "catch", "char", "char16_t", "char32_t", "class",
        "compl", "concept", "const", "const_cast", "continue", "decltype", "default",
        "delete", "do", "double", "dynamic_cast", "else", "enum", "explicit", "export",
        "extern", "false", "float", "for", "friend", "goto", "if", "inline", "int",
        "long", "mutable", "namespace", "new", "noexcept", "not", "not_eq", "nullptr",
        "operator", "or", "or_eq", "private", "protected", "public", "register",
        "reinterpret_cast", "return", "short", "signed", "sizeof", "static", "static_assert",
        "static_cast", "struct", "switch", "template", "this", "thread_local", "throw",
        "true", "try", "typedef", "typeid", "typename", "union", "unsigned", "using",
        "virtual", "void", "volatile", "wchar_t", "while", "xor", "xor_eq"
    ],
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


def extract_basic_features(file_path, lang):
    with open(file_path, "r", errors="ignore") as file:
        content = file.read()
        num_lines = content.count("\n")
        num_functions = len(
    re.findall(
        r"""
        # Python function definitions
        \bdef\b\s+\w+\s*\(|
        
        # JavaScript function definitions
        \bfunction\b\s*\w*\s*\(|
        
        # C, C++, Java, and C# function definitions
        \b(?:void|int|float|double|char|short|long|bool|String|string)\b\s+\w+\s*\(|
        
        # Go function definitions
        \bfunc\b\s+\w+\s*\(|
        
        # PHP function definitions
        \bfunction\b\s+\w+\s*\(|
        
        # Ruby method definitions
        \bdef\b\s+\w+\s*\(|
        
        # Kotlin function definitions
        \bfun\b\s+\w+\s*\(|
        
        # Swift function definitions
        \bfunc\b\s+\w+\s*\(
        """,
        content,
        re.VERBOSE
    )
)
        num_comments = len( re.findall(
        r"""
        # Single-line comments:
        # Python, Ruby, Perl
        \#.*|
        
        # C, C++, Java, JavaScript, C#, PHP, Go, Swift, Kotlin
        //.*|
        
        # Multi-line comments:
        # C, C++, Java, JavaScript, C#, PHP, Go, Swift, Kotlin
        /\*[\s\S]*?\*/|
        
        # Python multi-line comments (using triple quotes)
        # This captures both '''and '' '' '' as used in Python docstrings
        (?:\"\"\"[\s\S]*?\"\"\")|(?:\'\'\'[\s\S]*?\'\'\')
        """,
        content,
        re.DOTALL
    )
)
    
        num_keywords = sum(
            content.count(keyword) for keyword in LANGUAGE_KEYWORDS.get(lang, [])
        )

    features = {
        "num_lines": num_lines,
        "num_functions": num_functions,
        "num_comments": num_comments,
        "num_keywords": num_keywords,
    }
    return features


def get_pylint_metrics(file_path):
    try:
        result = subprocess.run(
            ["pylint", file_path, "-f", "json"],
            capture_output=True,
            text=True,
            check=True,
        )
        pylint_output = json.loads(result.stdout)
        metrics = {
            "num_errors": sum(
                1 for issue in pylint_output if issue.get("type") == "error"
            ),
            "num_warnings": sum(
                1 for issue in pylint_output if issue.get("type") == "warning"
            ),
        }
    except subprocess.CalledProcessError as e:
        logging.warning(f"Pylint failed for {file_path}: {e}")
        metrics = {"num_errors": 0, "num_warnings": 0}
    return metrics


def get_checkstyle_metrics(file_path):
    command = [r"C:/Program Files/Java/jdk-17/bin/java", "-jar", r"C:/Checkstyle/checkstyle-8.29-all.jar", "-c", "google_checks.xml", file_path, "-f", "xml"]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )
        root = ET.fromstring(result.stdout)
        errors = sum(1 for _ in root.findall(".//error"))
        warnings = sum(1 for _ in root.findall(".//warning"))
        metrics = {"num_errors": errors, "num_warnings": warnings}
    except subprocess.CalledProcessError as e:
        logging.warning(f"Checkstyle failed for {file_path}: {e}")
        metrics = {"num_errors": 0, "num_warnings": 0}
    return metrics


def get_eslint_metrics(file_path):
    try:
        result = subprocess.run(
            [r"C:/Users/user/AppData/Roaming/npm/eslint.cmd", file_path, "-f", "json"],
            capture_output=True,
            text=True,
            check=True,
        )
        eslint_output = json.loads(result.stdout)
        num_errors = 0
        num_warnings = 0
        for result in eslint_output:
            num_errors += sum(
                1 for message in result.get("messages", []) if message["severity"] == 2
            )
            num_warnings += sum(
                1 for message in result.get("messages", []) if message["severity"] == 1
            )

        metrics = {"num_errors": num_errors, "num_warnings": num_warnings}
    except subprocess.CalledProcessError as e:
        logging.warning(f"ESLint failed for {file_path}: {e}")
        metrics = {"num_errors": 0, "num_warnings": 0}
    return metrics


def get_golint_metrics(file_path):
    try:
        result = subprocess.run(["golint", file_path], capture_output=True, text=True)
        num_issues = len(result.stdout.splitlines())
        metrics = {"num_issues": num_issues}
    except subprocess.CalledProcessError as e:
        logging.warning(f"Golint failed for {file_path}: {e}")
        metrics = {"num_issues": 0}
    return metrics


def get_phpcs_metrics(file_path):
    try:
        result = subprocess.run(
            ["phpcs", "--standard=PSR2", file_path, "--report=json"],
            capture_output=True,
            text=True,
            check=True,
        )
        phpcs_output = json.loads(result.stdout)
        metrics = {
            "num_errors": phpcs_output["totals"]["errors"],
            "num_warnings": phpcs_output["totals"]["warnings"],
        }
    except subprocess.CalledProcessError as e:
        logging.warning(f"PHP_CodeSniffer failed for {file_path}: {e}")
        metrics = {"num_errors": 0, "num_warnings": 0}
    return metrics


def get_cppcheck_metrics(file_path):
    try:
        result = subprocess.run(
            [r"C:/Program Files/Cppcheck/cppcheck.exe", "--enable=all", "--xml", file_path],
            capture_output=True,
            text=True,
            check=True,
        )
        if result.stdout.strip():
             root = ET.fromstring(result.stdout)
             num_errors = len(root.findall(".//error"))
             metrics = {"num_errors": num_errors, "num_warnings": 0}
        else:
            print(f"Cppcheck returned empty or invalid output for {file_path}")
    except (subprocess.CalledProcessError, ET.ParseError) as e:
        logging.warning(f"Cppcheck failed for {file_path}: {e}")
        print(f"XML parsing failed for {file_path}: {e}")
        metrics = {"num_errors": 0, "num_warnings": 0}
    return metrics


def extract_features_and_labels(repo_path):
    data = []
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            file_path = os.path.join(root, file)
            logging.info(f"Processing file: {file_path}")

            lang = file.split(".")[-1]
            features = extract_basic_features(file_path, lang)

            if file.endswith(".py"):
                pylint_metrics = get_pylint_metrics(file_path)
                features.update(pylint_metrics)
            elif file.endswith(".java"):
                checkstyle_metrics = get_checkstyle_metrics(file_path)
                features.update(checkstyle_metrics)
            elif file.endswith(".js") or file.endswith(".ts"):
                eslint_metrics = get_eslint_metrics(file_path)
                features.update(eslint_metrics)
            elif file.endswith(".go"):
                golint_metrics = get_golint_metrics(file_path)
                features.update(golint_metrics)
            elif file.endswith(".php"):
                phpcs_metrics = get_phpcs_metrics(file_path)
                features.update(phpcs_metrics)
            elif file.endswith(".cpp") or file.endswith(".cxx") or file.endswith(".cc"):
                cppcheck_metrics = get_cppcheck_metrics(file_path)
                features.update(cppcheck_metrics)

            labels = label_issue_or_pr(
                {
                    "title": file_path,
                    "body": open(file_path, "r", errors="ignore").read(),
                }
            )
            features["labels"] = labels
            data.append(features)
            logging.info(f"Extracted features for {file_path}: {features}")

    return data


def compute_complexity_metrics(file_path):
    try:
        with open(file_path, "r", errors="ignore") as f:
            code = f.read()
        # Compute cyclomatic complexity using Radon
        cc_results = radon_complexity.cc_visit(code)
        avg_complexity = (
            sum(block.complexity for block in cc_results) / len(cc_results)
            if cc_results
            else 0
        )
        return {"cyclomatic_complexity": avg_complexity}
    except Exception as e:
        logging.error(f"Error computing complexity for {file_path}: {e}")
        return {"cyclomatic_complexity": 0}


def analyze_import_statements(file_path):
    try:
        with open(file_path, "r", errors="ignore") as f:
            tree = ast.parse(f.read(), filename=file_path)
        imports = [
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom)
        ]
        num_imports = len(imports)
        return {"num_imports": num_imports}
    except Exception as e:
        logging.error(f"Error analyzing imports for {file_path}: {e}")
        return {"num_imports": 0}


def evaluate_model(model, X_train, X_test, y_train, y_test):
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=1)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=1)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=1)
    roc_auc = roc_auc_score(
        y_test, model.predict_proba(X_test), multi_class="ovo", average="weighted"
    )

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
        "model": model.__class__.__name__,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "training_time": training_time,
        "prediction_time": prediction_time,
        "interpretability": evaluate_interpretability(model),
        "y_pred": y_pred,
        "y_test": y_test,
        "probs": model.predict_proba(X_test),
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
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix for {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"confusion_matrix_{model_name}.png")
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
    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label="ROC curve of class {0} (area = {1:0.2f})" "".format(i, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Receiver Operating Characteristic for {model_name}")
    plt.legend(loc="lower right")
    plt.savefig(f"roc_curve_{model_name}.png")
    plt.close()


def plot_precision_recall_curve(y_test, probs, model_name):
    precision = {}
    recall = {}
    n_classes = len(set(y_test))
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(
            y_test, probs[:, i], pos_label=i
        )

    plt.figure()
    colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            recall[i],
            precision[i],
            color=color,
            lw=2,
            label="PR curve of class {0}" "".format(i),
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve for {model_name}")
    plt.legend(loc="lower left")
    plt.savefig(f"precision_recall_curve_{model_name}.png")
    plt.close()


def plot_metrics_comparison(results):
    df = pd.DataFrame(results)
    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "training_time",
        "prediction_time",
    ]
    df_melted = df.melt(id_vars=["model"], value_vars=metrics)

    plt.figure(figsize=(14, 7))
    sns.barplot(data=df_melted, x="variable", y="value", hue="model")
    plt.title("Model Performance Metrics Comparison")
    plt.xlabel("Metrics")
    plt.ylabel("Values")
    plt.xticks(rotation=45)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig("metrics_comparison.png")
    plt.close()


def plot_learning_curve(estimator, X, y, model_name):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
    )
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.plot(train_sizes, train_mean, "o-", color="r", label="Training score")
    plt.plot(train_sizes, test_mean, "o-", color="g", label="Cross-validation score")
    plt.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.1,
        color="r",
    )
    plt.fill_between(
        train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g"
    )
    plt.title(f"Learning Curve for {model_name}")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.savefig(f"learning_curve_{model_name}.png")
    plt.close()


REPOSITORIES = [
    "riskmanagement",
    "risk_complaince",
    "Django-Website-master",
    "Tic-Tac-Toe-Game-using-Cpp-main",
]


def process_repository(repo):
    repo_path = os.path.join(CLONE_DIR, repo)
    logging.info(f"Analyzing repository {repo_path}...")
    repo_data = extract_features_and_labels(repo_path)
    for record in repo_data:
        record["repository"] = repo
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
    logging.info(
        df.head()
    )  # Debugging: print the first few rows to ensure labels are present
    df.to_csv("features_and_labels.csv", index=False)
    logging.info("Data saved to features_and_labels.csv")

    # Prepare features and labels
    if "labels" not in df.columns:
        logging.error("Error: 'labels' column not found in DataFrame.")
        return

    X = df.drop(["labels", "repository"], axis=1)
    y = df["labels"].astype("category").cat.codes

    # Impute missing values
    imputer = SimpleImputer(
        strategy="mean"
    )  # You can change the strategy to 'median', 'most_frequent', etc.
    X_imputed = imputer.fit_transform(X)

    # Feature selection
    X_new = SelectKBest(f_classif, k="all").fit_transform(X_imputed, y)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_new, y, test_size=0.2, random_state=42
    )

    # Check class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    class_counts = dict(zip(unique, counts))
    logging.info(f"Class distribution in y_train: {class_counts}")

    # Adjust k_neighbors for SMOTE
    min_class_size = min(class_counts.values())
    k_neighbors = min(
        5, min_class_size - 1
    )  # Ensure k_neighbors is smaller than the smallest class size

    # Apply SMOTE to the training data
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Define models and hyperparameters
    models = {
        "RandomForest": RandomForestClassifier(),
        "SVM": SVC(probability=True),
        "NeuralNetwork": MLPClassifier(max_iter=1000),
    }

    param_grids = {
        "RandomForest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
        },
        "SVM": {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"],
        },
        "NeuralNetwork": {
            "hidden_layer_sizes": [(100,), (100, 50)],
            "activation": ["relu", "tanh"],
            "solver": ["adam", "sgd"],
        },
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
        cm = confusion_matrix(y_test, result["y_pred"])
        plot_confusion_matrix(cm, name)

        # Plot ROC Curve
        plot_roc_curve(y_test, result["probs"], name)

        # Plot Precision-Recall Curve
        plot_precision_recall_curve(y_test, result["probs"], name)

        # Plot Learning Curve
        plot_learning_curve(best_model, X, y, name)

    # Ensemble Methods - Stacking
    estimators = [
        ("rf", RandomForestClassifier(n_estimators=100)),
        ("svm", SVC(kernel="linear", probability=True)),
        ("mlp", MLPClassifier(max_iter=500)),
    ]

    stacking_model = StackingClassifier(
        estimators=estimators, final_estimator=LogisticRegression()
    )

    logging.info("Training Stacking Model...")
    result = evaluate_model(stacking_model, X_train_res, X_test, y_train_res, y_test)
    results.append(result)

    # Plot Confusion Matrix for Stacking Model
    cm = confusion_matrix(y_test, result["y_pred"])
    plot_confusion_matrix(cm, "Stacking")

    # Plot ROC Curve for Stacking Model
    plot_roc_curve(y_test, result["probs"], "Stacking")

    # Plot Precision-Recall Curve for Stacking Model
    plot_precision_recall_curve(y_test, result["probs"], "Stacking")

    # Plot Learning Curve for Stacking Model
    plot_learning_curve(stacking_model, X, y, "Stacking")

    # Save results to a DataFrame and output
    results_df = pd.DataFrame(results)
    logging.info(results_df)
    results_df.to_csv("model_performance_comparison.csv", index=False)
    logging.info(
        "Model performance comparison saved to model_performance_comparison.csv"
    )

    # Plot comparison of metrics
    plot_metrics_comparison(results)


if __name__ == "__main__":
    main()
