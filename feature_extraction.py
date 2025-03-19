import base64
import json
import os
import re
import subprocess
import tempfile
import pandas as pd
import logging
import radon.complexity as radon_complexity
from textblob import TextBlob

logging.basicConfig(level=logging.DEBUG)
# Assuming JSON format
cache_directory = "github_cache"  # Replace with your actual path
# Additional file extensions mapped to their respective languages
extension_language_map = {
    # [extension mappings]
    '.py': 'Python',
    '.java': 'Java',
    '.js': 'JavaScript',
    '.cpp': 'C++',
    '.c': 'C',
    '.go': 'Go',
    '.php': 'PHP',
    '.rb': 'Ruby',
    '.cs': 'C#',
    '.ts': 'TypeScript',
    '.html': 'HTML',
    '.css': 'CSS',
    '.xml': 'XML',
    '.json': 'JSON',
    '.sh': 'Shell',
    '.bat': 'Batch',
    '.rs': 'Rust',
    '.swift': 'Swift',
    '.kt': 'Kotlin',
    '.m': 'Objective-C',
    '.class': 'Java Bytecode',
    '.o': 'Object File',
    '.dll': 'Windows DLL',
    '.exe': 'Executable',
}

# Function to run security checks based on language
def run_security_checks(file_content, language):
    print("language: ", language)
    if language == 'python':
        return run_bandit_security_checks(file_content)
    elif language == 'javascript':
        return run_eslint_security_checks(file_content)
    elif language == 'java':
        return run_spotbugs_security_checks(file_content)
    elif language == 'cpp':
        return run_cppcheck_security_checks(file_content)
    elif language == 'go':
        return run_gosec_security_checks(file_content)
    elif language == 'ruby':
        return run_brakeman_security_checks(file_content)
    else:
        return None

# Security check functions for different languages, as previously defined...
def run_bandit_security_checks(file_content):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
        temp_filename = temp_file.name
        temp_file.write(file_content)
        temp_file.flush()  # Ensure all data is written to the file

    try:
        # Run Bandit on the temporary file
        result = subprocess.run(['bandit', '-r', temp_filename], capture_output=True, text=True)
        print("Bandit Result: ", result.stdout)
    except Exception as e:
        print(f"Error running Bandit: {e}")
    finally:
        # Ensure the temporary file is deleted after processing
        try:
            os.remove(temp_filename)
        except Exception as e:
            print(f"Error removing temp file {temp_filename}: {e}")
    
    return result.stdout if 'result' in locals() else None
# Function to run ESLint for JavaScript/TypeScript
def run_eslint_security_checks(file_content):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as temp_file:
        temp_filename = temp_file.name
        temp_file.write(file_content)
        temp_file.flush()  # Ensure all data is written to the file

    try:
        # Run Elsint on the temporary file
        result = subprocess.run(['C:/Users/user/AppData/Roaming/npm/eslint.cmd', temp_filename], capture_output=True, text=True)
        print("Eslint Result: ", result.stdout)
    except Exception as e:
        print(f"Error running Eslint: {e}")
    finally:
        # Ensure the temporary file is deleted after processing
        try:
            os.remove(temp_filename)
        except Exception as e:
            print(f"Error removing temp file {temp_filename}: {e}")
    
    return result.stdout if 'result' in locals() else None
# Function to run SpotBugs for Java
def run_spotbugs_security_checks(file_content):
    # Create a temporary file with the given Java code
    with tempfile.NamedTemporaryFile(mode='w', suffix='.java', delete=False) as temp_file:
        temp_filename = temp_file.name
        temp_file.write(file_content)
        temp_file.flush()  # Ensure all data is written to the file

    try:
        # Debug: Check if the temporary file exists and is accessible
        print(f"Temporary file created at: {temp_filename}")
        if not os.path.exists(temp_filename):
            print(f"Error: Temporary file {temp_filename} does not exist!")
            return None

        # Run SpotBugs on the temporary file
        result = subprocess.run(
            ['C:/spotbugs/bin/spotbugs.bat', '-textui', '-high', '-effort:max', temp_filename],
            capture_output=True, text=True
        )

        # Debug: Print the standard output and standard error
        print("SpotBugs Stdout:\n", result.stdout)
        print("SpotBugs Stderr:\n", result.stderr)

        # Check the return code to see if the command was successful
        if result.returncode != 0:
            print(f"SpotBugs encountered an error (return code {result.returncode}). Check stderr for details.")
    except Exception as e:
        print(f"Error running SpotBugs: {e}")
    finally:
        # Ensure the temporary file is deleted after processing
        try:
            os.remove(temp_filename)
            print(f"Temporary file {temp_filename} deleted.")
        except Exception as e:
            print(f"Error removing temp file {temp_filename}: {e}")
    
    # Return the SpotBugs result if available
    return result.stdout if 'result' in locals() else None
# Function to run Cppcheck for C/C++
def run_cppcheck_security_checks(file_content):

    with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as temp_file:
        temp_filename = temp_file.name
        temp_file.write(file_content)
        temp_file.flush()  # Ensure all data is written to the file

    try:
        # Run Cppcheck on the temporary file
        result = subprocess.run(['C:/Program Files/Cppcheck/cppcheck.exe', '--enable=all', temp_filename], capture_output=True, text=True)
        print("Cppcheck Result: ", result.stdout)
    except Exception as e:
        print(f"Error running Cppcheck: {e}")
    finally:
        # Ensure the temporary file is deleted after processing
        try:
            os.remove(temp_filename)
        except Exception as e:
            print(f"Error removing temp file {temp_filename}: {e}")
    
    return result.stdout if 'result' in locals() else None

# Function to run Gosec for Go
def run_gosec_security_checks(file_content):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.go', delete=False) as temp_file:
        temp_filename = temp_file.name
        temp_file.write(file_content)
        temp_file.flush()  # Ensure all data is written to the file

    try:
        # Run Gosec on the temporary file
        result = subprocess.run(['gosec', temp_filename], capture_output=True, text=True)
        print("Gosec Result: ", result.stdout)
    except Exception as e:
        print(f"Error running Gosec: {e}")
    finally:
        # Ensure the temporary file is deleted after processing
        try:
            os.remove(temp_filename)
        except Exception as e:
            print(f"Error removing temp file {temp_filename}: {e}")
    
    return result.stdout if 'result' in locals() else None

# Function to run Brakeman for Ruby
def run_brakeman_security_checks(file_content):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.rb', delete=False) as temp_file:
        temp_filename = temp_file.name
        temp_file.write(file_content)
        temp_file.flush()  # Ensure all data is written to the file

    try:
        # Run Brakeman on the temporary file
        result = subprocess.run(['brakeman', temp_filename], capture_output=True, text=True)
        print("Brakeman Result: ", result.stdout)
    except Exception as e:
        print(f"Error running Brakeman: {e}")
    finally:
        # Ensure the temporary file is deleted after processing
        try:
            os.remove(temp_filename)
        except Exception as e:
            print(f"Error removing temp file {temp_filename}: {e}")
    
    return result.stdout if 'result' in locals() else None
# Function to calculate sentiment of review comments
def calculate_sentiment(comment):
    return TextBlob(comment).sentiment.polarity

def analyze_code_quality(file_content, language):
    issues = []
    if language == 'python':
        if 'print(' in file_content:
            issues.append("Print statements found.")
        if re.search(r'#.*\b(if|for|while|return|class|def)\b', file_content):
            issues.append("Commented-out code found.")
        long_lines = [line for line in file_content.splitlines() if len(line) > 79]
        if long_lines:
            issues.append(f"{len(long_lines)} lines exceed 79 characters.")
    elif language == 'javascript':
        if 'console.log(' in file_content:
            issues.append("Console.log statements found.")
        if '/*' in file_content or '*/' in file_content:
            issues.append("Commented-out code found.")
        long_lines = [line for line in file_content.splitlines() if len(line) > 80]
        if long_lines:
            issues.append(f"{len(long_lines)} lines exceed 80 characters.")
    elif language == 'java':
        if 'System.out.println(' in file_content:
            issues.append("System.out.println statements found.")
        if '/*' in file_content or '*/' in file_content:
            issues.append("Commented-out code found.")
        long_lines = [line for line in file_content.splitlines() if len(line) > 100]
        if long_lines:
            issues.append(f"{len(long_lines)} lines exceed 100 characters.")
    elif language == 'cpp':
        if '#include' in file_content:
            issues.append("Header includes found.")
        if '/*' in file_content or '*/' in file_content:
            issues.append("Commented-out code found.")
        long_lines = [line for line in file_content.splitlines() if len(line) > 100]
        if long_lines:
            issues.append(f"{len(long_lines)} lines exceed 100 characters.")
    elif language == 'go':
        if 'fmt.Println(' in file_content:
            issues.append("fmt.Println statements found.")
        if '/*' in file_content or '*/' in file_content:
            issues.append("Commented-out code found.")
        long_lines = [line for line in file_content.splitlines() if len(line) > 80]
        if long_lines:
            issues.append(f"{len(long_lines)} lines exceed 80 characters.")
    elif language == 'ruby':
        if 'puts ' in file_content:
            issues.append("Put statements found.")
        if '# ' in file_content:
            issues.append("Commented-out code found.")
        long_lines = [line for line in file_content.splitlines() if len(line) > 80]
        if long_lines:
            issues.append(f"{len(long_lines)} lines exceed 80 characters.")
    return issues if issues else ["No major code quality issues found."]

  
def detect_language(file_name):
    ext = file_name.split('.')[-1]
    
    return extension_language_map.get(f'.{ext}', None)
def calculate_complexity(file_content, file_extension):
    if file_extension.lower() != '.py':
        print("Complexity calculation is only supported for Python files.")
        return None
    
    try:
        complexity = radon_complexity.cc_visit(file_content)
        return sum([item.complexity for item in complexity]) / len(complexity) if complexity else 0
    except Exception as e:
        print(f"Error calculating complexity: {e}")
        return None
def extract_features(pr_data, repo_name):
    pr_complexity_scores = []
    pr_security_issues = []
    pr_code_quality_issues = []

    # Adjust according to the actual structure
    files = pr_data.get('files', [])  # This may need adjustment
    if isinstance(files, list):  # Ensure files is a list
        for file in files:
            file_path = os.path.join(repo_name, file.get('filename', ''))
            file_content, _ = process_local_file(file_path)
            print("file_content: ", file_content)
            if file_content:
                language = detect_language(file.get('filename', ''))
                if language:
                    security_issues = run_security_checks(file_content, language)
                    if security_issues:
                        pr_security_issues.append(security_issues)

                    code_quality_issues = analyze_code_quality(file_content, language)
                    if code_quality_issues:
                        pr_code_quality_issues.append("\n".join(code_quality_issues))

                    # Avoid complexity calculation for unknown or binary files
                    complexity_score = calculate_complexity(file_content, file_path)
                    if complexity_score is not None:
                        pr_complexity_scores.append(complexity_score)

    avg_complexity_score = sum(pr_complexity_scores) / len(pr_complexity_scores) if pr_complexity_scores else 0
    reviews = pr_data.get('reviews', [])
    
    pr_details = {
        "pr_id": pr_data.get('number'),
        "repository_name": repo_name,
        "title_length": len(pr_data.get('title', '')),
        "description_length": len(pr_data.get('body', '')) if pr_data.get('body') else 0,
        "files_changed": pr_data.get('changed_files', 0),
        "lines_added": pr_data.get('additions', 0),
        "lines_deleted": pr_data.get('deletions', 0),
        "complexity_score": avg_complexity_score,
        "time_to_merge": pr_data.get('time_to_merge'),
        "num_commits": pr_data.get('commits', 0),
        "is_bug_fix": 'bug' in pr_data.get('labels', []),
        "num_comments": pr_data.get('comments', 0) + pr_data.get('review_comments', 0),
        "num_reviewers": len(set([review.get('user') for review in reviews])),
        "merge_status": pr_data.get('merged', False),
        "author_experience": pr_data.get('author_experience', 0),
        "author_tenure": pr_data.get('author_tenure', 0),
        "author_contributions": pr_data.get('author_contributions', 0),
        "review_time": sum([review.get('review_time', 0) for review in reviews]),
        "num_approvals": len([review for review in reviews if review.get('state') == 'APPROVED']),
        "review_sentiment_score": sum([calculate_sentiment(comment.get('body', '')) for comment in pr_data.get('issue_comments', [])]),
        "security_issues": "\n".join(pr_security_issues),
        "code_quality_issues": "\n".join(pr_code_quality_issues)
    }
    
    return pr_details

def process_local_file(file_path):
    try:
        _, ext = file_path.lower().rsplit('.', 1)
        language = extension_language_map.get(f'.{ext}', 'Unknown')

        if ext in ['mo', 'exe', 'dll', 'o', 'class']:
            with open(file_path, 'rb') as f:
                binary_content = f.read()
            encoded_content = base64.b64encode(binary_content).decode('utf-8')
            return encoded_content, language
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                decoded_content = f.read()
            return decoded_content, language

    except UnicodeDecodeError as e:
        print(f"Error decoding {file_path}: {e}")
        return None, language
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, language
    
def save_to_csv(pr_details, csv_file):
    df = pd.DataFrame([pr_details])
    if not os.path.isfile(csv_file):
        df.to_csv(csv_file, index=False, mode='w', header=True, encoding='utf-8')
    else:
        df.to_csv(csv_file, index=False, mode='a', header=False, encoding='utf-8')

def process_repository_files(repo_path, csv_file):
    pr_features = []
    for root, dirs, files in os.walk(repo_path):
        for filename in files:
            if filename.endswith(".json"):
                file_path = os.path.join(root, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        pr_data = json.load(f)
                        if isinstance(pr_data, dict) and "number" in pr_data:
                            pr_feature = extract_features(pr_data, os.path.basename(repo_path))
                            save_to_csv(pr_feature, csv_file)
                            pr_features.append(pr_feature)
                        else:
                            print(f"Unexpected data format in {file_path}. Skipping file.")
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in {file_path}: {e}")
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
    return pr_features

def process_all_repositories(cache_directory, csv_file):
    all_pr_features = []
    for repo_name in os.listdir(cache_directory):
        repo_path = os.path.join(cache_directory, repo_name)
        if os.path.isdir(repo_path):
            print(f"Processing repository: {repo_name}")
            repo_pr_features = process_repository_files(repo_path, csv_file)
            if repo_pr_features:  # Ensure it's not None
                all_pr_features.extend(repo_pr_features)
    return all_pr_features



def main():
    csv_file_path = 'repo_analysis_results.csv'
    process_all_repositories(cache_directory, csv_file_path)

if __name__ == "__main__":
    main()
