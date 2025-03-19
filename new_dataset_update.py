


import base64
from datetime import datetime, timedelta
import json
import os
import re
import tempfile
import time
from github import Github, GithubException
import pandas as pd
from textblob import TextBlob
import concurrent.futures
import logging
import subprocess

logging.basicConfig(level=logging.DEBUG)
CACHE_DIR = "github_cache"
GITHUB_TOKENS = [
    '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',
    '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',
    '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',
    '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',
    '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',
    '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'
    # Add more tokens as needed
]
extension_language_map = {
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

# Function to run Bandit for Python

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

current_token_index = 0

def get_github_token():
    global current_token_index
    return GITHUB_TOKENS[current_token_index]

def switch_github_token():
    global current_token_index
    current_token_index = (current_token_index + 1) % len(GITHUB_TOKENS)
    print(f"Switched to token {current_token_index + 1} of {len(GITHUB_TOKENS)}")

def generate_cache_key(repo_name, pr_number=None):
    key = f"{repo_name}_pr_{pr_number}.json" if pr_number else f"{repo_name}.json"
    return os.path.join(CACHE_DIR, key)

def load_from_cache(cache_key):
    if not os.path.exists(cache_key):
        return None

    try:
        with open(cache_key, 'r') as cache_file:
            return json.load(cache_file)
    except json.JSONDecodeError:
        print(f"Warning: Cache file {cache_key} is corrupted or empty.")
        return None


def save_to_cache(data, cache_key):
    os.makedirs(os.path.dirname(cache_key), exist_ok=True)
    json_data = json.dumps(data, indent=4)
    with open(cache_key, 'w') as cache_file:
        cache_file.write(json_data)
        
def check_rate_limit(github_instance):
    rate_limit = github_instance.get_rate_limit().core
    return rate_limit.remaining, rate_limit.reset

def wait_for_rate_limit_reset(reset_time):
    now = datetime.utcnow()
    wait_time = (reset_time - now).total_seconds() + 10  # Add a 10-second buffer
    print(f"Rate limit exceeded. Waiting for {int(wait_time)} seconds.")
    time.sleep(wait_time)

def detect_language(file_name):
    ext = file_name.split('.')[-1]
    if ext in ['py']:
        return 'python'
    elif ext in ['js', 'ts']:
        return 'javascript'
    elif ext in ['java']:
        return 'java'
    elif ext in ['c', 'cpp', 'h']:
        return 'cpp'
    elif ext in ['go']:
        return 'go'
    elif ext in ['rb']:
        return 'ruby'
    else:
        return None

def fetch_with_retry(github_instance, url, max_retries=3, retry_delay=5):
    retries = 0
    while retries < max_retries:
        remaining, reset_time = check_rate_limit(github_instance)
        if remaining == 0:
            switch_github_token()
            github_instance = Github(get_github_token())
            remaining, reset_time = check_rate_limit(github_instance)
            if remaining == 0:
                wait_for_rate_limit_reset(reset_time)
        
        try:
            return github_instance.get_repo(url)
        except GithubException as e:
            if e.status == 403 and "rate limit" in str(e):
                switch_github_token()
                retries += 1
            else:
                raise e
        time.sleep(retry_delay)
    
    raise Exception("Max retries reached. Failed to fetch data.")
def process_file(file_path, repo):
    try:
        file_content = repo.get_contents(file_path)

        # Guess the file type based on its extension
        _, ext = file_path.lower().rsplit('.', 1)
        language = extension_language_map.get(f'.{ext}', 'Unknown')

        if ext in ['mo', 'exe', 'dll', 'o', 'class']:
            # Handle binary files
            binary_content = file_content.decoded_content
            print(f"Processing binary file: {file_path} ({language})")
            
            # Optionally, encode to Base64 for JSON serialization
            encoded_content = base64.b64encode(binary_content).decode('utf-8')
            
            # Skip JSON serialization if not required, or use encoded_content
            return encoded_content, language
        
        else:
            # Handle text files
            decoded_content = file_content.decoded_content.decode('utf-8')
            return decoded_content, language

    except UnicodeDecodeError as e:
        print(f"Error decoding {file_path}: {e}")
        return None, language
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, language

def get_file_content(github_instance,repo_name, file_path):
    
    cache_key = generate_cache_key(repo_name, file_path)
    cached_data = load_from_cache(cache_key)
    
    if cached_data:
        logging.info(f"Loaded file content from cache for file: {file_path}")
        return cached_data

    try:
        repo = github_instance.get_repo(repo_name)
        file_content = process_file(repo, file_path)
        save_to_cache(file_content, cache_key)  # Save file content to cache
        print("file_content: ", file_content)
        return file_content
    except (Exception, GithubException) as e:
        logging.error(f"Error fetching file content for {file_path}: {e}")
        return None
def process_pull_request(github_instance, pr, repo_name):
    try:
        pr_security_issues = []
        pr_code_quality_issues = []

        for file in pr.get_files():
            file_path = file.filename  # Ensure this variable is correctly set
            try:
                file_content = get_file_content(github_instance, repo_name, file_path)
            except Exception as e:
                logging.error(f"Error fetching content for file {file_path}: {e}")
                continue

            if file_content:
                language = detect_language(file.filename)
                if language:
                    security_issues = run_security_checks(file_content, language)
                    if security_issues:
                        pr_security_issues.append(security_issues)
                    
                    code_quality_issues = analyze_code_quality(file_content, language)
                    if code_quality_issues:
                        pr_code_quality_issues.append("\n".join(code_quality_issues))

        reviews = list(pr.get_reviews())
        
        pr_details = {
            "pr_id": pr.number,
            "repository_name": repo_name,
            "title_length": len(pr.title),
            "description_length": len(pr.body) if pr.body else 0,
            "files_changed": pr.changed_files,
            "lines_added": pr.additions,
            "lines_deleted": pr.deletions,
            "time_to_merge": (pr.merged_at - pr.created_at).total_seconds() / 3600 if pr.merged_at else None,
            "num_commits": pr.commits,
            "is_bug_fix": 'bug' in [label.name for label in pr.get_labels()],
            "num_comments": pr.comments + pr.review_comments,
            "num_reviewers": len(set([review.user.login for review in reviews])),
            "merge_status": pr.merged,
            "author_experience": len(github_instance.get_user(pr.user.login).get_repos()),
            "author_tenure": (pr.created_at - pr.user.created_at).days / 30,
            "author_contributions": pr.user.contributions,
            "review_time": sum([(review.submitted_at - pr.created_at).total_seconds() / 3600 for review in reviews if review.submitted_at]),
            "num_approvals": len([review for review in reviews if review.state == 'APPROVED']),
            "review_sentiment_score": sum([calculate_sentiment(comment.body) for comment in pr.get_issue_comments()]),
            "security_issues": "\n".join(pr_security_issues),
            "code_quality_issues": "\n".join(pr_code_quality_issues)
        }
        logging.info(f"Pull Request Details: {pr_details}")
        return pr_details

    except Exception as e:
        logging.error(f"Error processing PR {pr.number} in {repo_name}: {e}")
        return None

def check_if_csv_exists(file_path):
    if os.path.isfile(file_path):
        print(f"The file '{file_path}' already exists.")
        return True
    else:
        print(f"The file '{file_path}' does not exist.")
        return False

def append_to_csv(data, csv_path):
    df = pd.DataFrame(data)
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)

def process_all_repositories(github_instance, repos):
    pr_data = []
    for repo in repos:
        print(f"Processing repository: {repo['name']}")
        repository = fetch_with_retry(github_instance, repo['name'])
        print("repo_type: ",type(repository))
        pull_requests = repository.get_pulls(state='all')
        for pr in pull_requests:
            pr_details = process_pull_request(github_instance, pr, repo['name'])
            if pr_details:
                pr_data.append(pr_details)
    
    return pr_data

def main():
    print("I'm in main methods")
    github_instance = Github(get_github_token())
    repositories = [
          {"name":"django/django"},
          {"name": "microsoft/vscode"},
          {"name": "facebook/react"},
          {"name": "spring-projects/spring-framework"},
          {"name": "kubernetes/kubernetes"},
          {"name": 'opencv/opencv'}
    ]
    pr_data = []
    try:
        pr_details_list = process_all_repositories(github_instance, repositories)
        for pr_details in pr_details_list:
            pr_data.append(pr_details)
            print("pr_details: ", pr_details)
    except Exception as e:
        print(f"An error occurred: {e}")
    output_file = 'pull_requests_and_analysis.csv'
    print("before save pr_data: ", pr_data)
    if pr_data:
        df = pd.DataFrame(pr_data)
        print("DataFrame created successfully.")
        print(df.head())  # Check the first few rows of the DataFrame

       
        try:
            if check_if_csv_exists(output_file):
                append_to_csv(pr_data, output_file)
                print(f"Dataset updated to {output_file}")
            else:
                df.to_csv(output_file, index=False)
                print(f"Dataset saved to {output_file}")
        except Exception as e:
            print(f"Error saving to {output_file}: {e}")
    else:
        print("No data to save.")
if __name__ == "__main__":
    main()
