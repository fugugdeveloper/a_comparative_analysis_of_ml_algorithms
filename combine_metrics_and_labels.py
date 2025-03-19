import os

def extract_features_and_labels(repo_path):
    data = []
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if file.endswith(('.py', '.js', '.java', '.go', '.rb', '.php')):  # Add more extensions as needed
                file_path = os.path.join(root, file)
                features = extract_basic_features(file_path)
                if file.endswith('.py'):
                    pylint_metrics = get_pylint_metrics(file_path)
                    features.update(pylint_metrics)
                labels = label_issue_or_pr({'title': file_path, 'body': open(file_path, 'r', errors='ignore').read()})
                data.append(features + [labels])
    return data

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
