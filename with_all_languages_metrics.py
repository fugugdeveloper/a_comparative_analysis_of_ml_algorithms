import os
import re
import subprocess
import json
import logging
import spacy
from collections import defaultdict
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define keywords for Python, Java, JavaScript, Go, PHP, and C++
LANGUAGE_KEYWORDS = {
    'py': ['def', 'class', 'import', 'return', 'if', 'else', 'elif', 'try', 'except', 'for', 'while'],
    'java': ['class', 'public', 'private', 'protected', 'void', 'int', 'if', 'else', 'try', 'catch', 'for', 'while'],
    'js': ['function', 'class', 'const', 'let', 'var', 'if', 'else', 'try', 'catch', 'for', 'while'],
    'ts': ['function', 'class', 'const', 'let', 'var', 'if', 'else', 'try', 'catch', 'for', 'while'],
    'go': ['func', 'package', 'import', 'return', 'if', 'else', 'for'],
    'php': ['function', 'class', 'public', 'private', 'protected', 'return', 'if', 'else', 'for', 'while'],
    'cpp': ['int', 'void', 'class', 'public', 'private', 'protected', 'if', 'else', 'for', 'while', 'try', 'catch']
}
nlp = spacy.load("en_core_web_sm")

# Define keywords for different categories
LABEL_KEYWORDS = {
    'bug': [
        'bug', 'error', 'issue', 'fix', 'crash', 'fail', 'wrong', 'incorrect', 'problem'
    ],
    'feature': [
        'feature', 'new', 'add', 'introduce', 'implement', 'provide', 'create', 'integrate'
    ],
    'enhancement': [
        'enhancement', 'improve', 'upgrade', 'optimize', 'better', 'expand', 'develop', 'refine'
    ],
    'documentation': [
        'doc', 'documentation', 'comment', 'guide', 'help', 'manual', 'describe', 'explain'
    ]
}

def label_issue_or_pr(issue):
    # Combine title and body for NLP processing
    content = (issue['title'] + " " + issue['body']).lower()

    # Process content with spaCy
    doc = nlp(content)

    # Dictionary to store matched keyword counts for each label
    label_counts = defaultdict(int)

    # Count keyword occurrences using spaCy tokens
    for token in doc:
        for label, keywords in LABEL_KEYWORDS.items():
            if token.lemma_ in keywords:
                label_counts[label] += 1

    # Incorporate additional NLP features (e.g., sentence similarity)
    for sent in doc.sents:
        for label, keywords in LABEL_KEYWORDS.items():
            if any(keyword in sent.text.lower() for keyword in keywords):
                label_counts[label] += 1

    # Determine the label with the highest count
    if label_counts:
        most_common_label = max(label_counts, key=label_counts.get)
        if label_counts[most_common_label] > 0:
            return most_common_label

    # Default to 'other' if no keywords match
    return 'other'

# Example usage
issue = {
    'title': 'Fix crashing issue in user login feature',
    'body': 'The login feature crashes when the user enters incorrect credentials. This bug needs urgent attention.'
}

label = label_issue_or_pr(issue)
print(f"Label: {label}")
def extract_basic_features(file_path, lang):
    with open(file_path, 'r', errors='ignore') as file:
        content = file.read()
        num_lines = content.count('\n')
        num_functions = len(re.findall(r'\bdef\b|\bfunction\b|\bfunc\b|\bvoid\b|\bint\b|\bString\b|\bchar\b|\bstring\b', content))
        num_comments = len(re.findall(r'#.*|//.*|/\*.*?\*/', content, re.DOTALL))
        num_keywords = sum(content.count(keyword) for keyword in LANGUAGE_KEYWORDS.get(lang, []))

    features = {
        'num_lines': num_lines,
        'num_functions': num_functions,
        'num_comments': num_comments,
        'num_keywords': num_keywords,
    }
    return features

def get_pylint_metrics(file_path):
    try:
        result = subprocess.run(['pylint', file_path, '-f', 'json'], capture_output=True, text=True, check=True)
        pylint_output = json.loads(result.stdout)
        metrics = {
            'num_errors': sum(1 for issue in pylint_output if issue.get('type') == 'error'),
            'num_warnings': sum(1 for issue in pylint_output if issue.get('type') == 'warning')
        }
    except subprocess.CalledProcessError as e:
        logging.warning(f"Pylint failed for {file_path}: {e}")
        metrics = {'num_errors': 0, 'num_warnings': 0}
    return metrics
import xml.etree.ElementTree as ET

def get_checkstyle_metrics(file_path):
    try:
        result = subprocess.run(['checkstyle', '-c', '/google_checks.xml', file_path, '-f', 'xml'], capture_output=True, text=True, check=True)
        root = ET.fromstring(result.stdout)
        errors = sum(1 for _ in root.findall('.//error'))
        warnings = sum(1 for _ in root.findall('.//warning'))
        metrics = {
            'num_errors': errors,
            'num_warnings': warnings
        }
    except subprocess.CalledProcessError as e:
        logging.warning(f"Checkstyle failed for {file_path}: {e}")
        metrics = {'num_errors': 0, 'num_warnings': 0}
    return metrics
def get_eslint_metrics(file_path):
    try:
        result = subprocess.run(['eslint', file_path, '-f', 'json'], capture_output=True, text=True, check=True)
        eslint_output = json.loads(result.stdout)
        num_errors = 0
        num_warnings = 0
        for result in eslint_output:
            num_errors += sum(1 for message in result.get('messages', []) if message['severity'] == 2)
            num_warnings += sum(1 for message in result.get('messages', []) if message['severity'] == 1)

        metrics = {
            'num_errors': num_errors,
            'num_warnings': num_warnings
        }
    except subprocess.CalledProcessError as e:
        logging.warning(f"ESLint failed for {file_path}: {e}")
        metrics = {'num_errors': 0, 'num_warnings': 0}
    return metrics
def get_golint_metrics(file_path):
    try:
        result = subprocess.run(['golint', file_path], capture_output=True, text=True)
        num_issues = len(result.stdout.splitlines())
        metrics = {
            'num_issues': num_issues
        }
    except subprocess.CalledProcessError as e:
        logging.warning(f"Golint failed for {file_path}: {e}")
        metrics = {'num_issues': 0}
    return metrics
def get_phpcs_metrics(file_path):
    try:
        result = subprocess.run(['phpcs', '--standard=PSR2', file_path, '--report=json'], capture_output=True, text=True, check=True)
        phpcs_output = json.loads(result.stdout)
        metrics = {
            'num_errors': phpcs_output['totals']['errors'],
            'num_warnings': phpcs_output['totals']['warnings']
        }
    except subprocess.CalledProcessError as e:
        logging.warning(f"PHP_CodeSniffer failed for {file_path}: {e}")
        metrics = {'num_errors': 0, 'num_warnings': 0}
    return metrics
def get_cppcheck_metrics(file_path):
    try:
        result = subprocess.run(['cppcheck', '--enable=all', '--xml', file_path], capture_output=True, text=True, check=True)
        root = ET.fromstring(result.stdout)
        num_errors = len(root.findall('.//error'))
        metrics = {
            'num_errors': num_errors,
            'num_warnings': 0  # Cppcheck doesn't separately report warnings, but you can parse as needed
        }
    except subprocess.CalledProcessError as e:
        logging.warning(f"Cppcheck failed for {file_path}: {e}")
        metrics = {'num_errors': 0, 'num_warnings': 0}
    return metrics
def extract_features_and_labels(repo_path):
    data = []
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            file_path = os.path.join(root, file)
            logging.info(f"Processing file: {file_path}")

            lang = file.split('.')[-1]
            # Extract basic features
            features = extract_basic_features(file_path, lang)

            if file.endswith('.py'):
                pylint_metrics = get_pylint_metrics(file_path)
                features.update(pylint_metrics)
            elif file.endswith('.java'):
                checkstyle_metrics = get_checkstyle_metrics(file_path)
                features.update(checkstyle_metrics)
            elif file.endswith('.js') or file.endswith('.ts'):
                eslint_metrics = get_eslint_metrics(file_path)
                features.update(eslint_metrics)
            elif file.endswith('.go'):
                golint_metrics = get_golint_metrics(file_path)
                features.update(golint_metrics)
            elif file.endswith('.php'):
                phpcs_metrics = get_phpcs_metrics(file_path)
                features.update(phpcs_metrics)
            elif file.endswith('.cpp') or file.endswith('.cxx') or file.endswith('.cc'):
                cppcheck_metrics = get_cppcheck_metrics(file_path)
                features.update(cppcheck_metrics)

            labels = label_issue_or_pr({'title': file_path, 'body': open(file_path, 'r', errors='ignore').read()})
            features['labels'] = labels
            data.append(features)
            logging.info(f"Extracted features for {file_path}: {features}")
    
    return data


def main():
    repo_path = 'path_to_your_repository'  # Replace with your repo path
    data = extract_features_and_labels(repo_path)
    # Save data to a file or process further
    with open('features_and_labels.json', 'w') as f:
        json.dump(data, f)
    logging.info(f"Data saved to features_and_labels.json")

if __name__ == "__main__":
    main()
    
REPOSITORIES = [
    "facebook/react",
    "microsoft/vscode",
    "django/django",
    "spring-projects/spring-framework",
    "kubernetes/kubernetes",
    'opencv/opencv'
]

