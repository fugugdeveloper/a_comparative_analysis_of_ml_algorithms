import base64
import json
import mimetypes
import os
import re
import tempfile
import time
from github import Github, GithubException
import pandas as pd
import requests
from textblob import TextBlob
import concurrent.futures
import logging
import subprocess

logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(encoding='utf-8')

ACCESS_TOKENS = [
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
# Authenticate to GitHub
current_token_index = 0
g = Github(ACCESS_TOKENS[current_token_index], timeout=120)
rate_limit = g.get_rate_limit()
print("Rate_Limit: ",rate_limit)
# List of repositories you want to analyze
CACHE_DIR = "github_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
repositories = [
    "django/django",
    "facebook/react",
    "microsoft/vscode",
    "spring-projects/spring-framework",
    "kubernetes/kubernetes",
    'opencv/opencv'
]
def switch_token():
    global current_token_index, g
    current_token_index += 1
    if current_token_index < len(ACCESS_TOKENS):
        g = Github(ACCESS_TOKENS[current_token_index], timeout=120)
        logging.info(f"Switched to token {current_token_index + 1}")
    else:
        logging.error("All tokens have been exhausted.")
        raise Exception("Rate limit reached on all tokens.")

 # Add more repository names as needed
def handle_rate_limit():
    rate_limit = g.get_rate_limit()
    if rate_limit.core.remaining == 0:
        logging.info("Rate limit exceeded. Switching to the next token.")
        switch_token()  # Switch to the next token

# Function to detect the programming language based on file extension
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

def generate_cache_key(repo_name, pr_number=None):
    key = f"{repo_name}_pr_{pr_number}.json" if pr_number else f"{repo_name}.json"
    return os.path.join(CACHE_DIR, key)

def save_to_cache(data, cache_key):
    # Ensure that the directory exists
    os.makedirs(os.path.dirname(cache_key), exist_ok=True)
    cache_dir = os.path.dirname(cache_key)
    os.makedirs(cache_dir, exist_ok=True)
    # Convert the dictionary to a JSON string
    json_data = json.dumps(data, indent=4)
    
    # Save the JSON string to the cache file
    with open(cache_key, 'w') as cache_file:
        cache_file.write(json_data)
        
def load_from_cache(cache_key):
    if not os.path.exists(cache_key):
        return None

    try:
        with open(cache_key, 'r') as cache_file:
            return json.load(cache_file)
    except json.JSONDecodeError:
        print(f"Warning: Cache file {cache_key} is corrupted or empty.")
        return None

def fetch_with_retry(repo_name, retries=5, backoff=2):
    cache_key = generate_cache_key(repo_name)
    cached_data = load_from_cache(cache_key)

    if cached_data:
        logging.info(f"Loaded data from cache for repository: {repo_name}")
        return cached_data

    for i in range(retries):
        try:
            repo = g.get_repo(repo_name)
            save_to_cache(repo.raw_data, cache_key)  # Save repo data to cache
            return repo
        except GithubException as e:
            if e.status == 403 and 'rate limit' in str(e):
                handle_rate_limit()
                continue  # Retry after handling rate limit
            elif e.status == 403:
                logging.warning(f"Access forbidden for repository: {repo_name}. Check your token and permissions.")
                return None
            elif e.status == 404:
                logging.warning(f"Repository not found: {repo_name}.")
                return None
            elif isinstance(e, requests.exceptions.ReadTimeout):
                if i < retries - 1:
                    sleep_time = backoff * (2 ** i)
                    logging.warning(f"Request timed out: {e}. Retrying in {sleep_time} seconds.")
                    time.sleep(sleep_time)
                else:
                    logging.error(f"Max retries exceeded: {e}")
                    return None
            else:
                logging.error(f"Error accessing repository {repo_name}: {e}")
                return None
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

def get_file_content(repo, file_path):
    cache_key = generate_cache_key(repo.full_name, file_path)
    cached_data = load_from_cache(cache_key)
    
    if cached_data:
        logging.info(f"Loaded file content from cache for file: {file_path}")
        return cached_data

    try:
        file_content = process_file(file_path, repo)
        save_to_cache(file_content, cache_key)  # Save file content to cache
        print("file_content: ", file_content)
        return file_content
    except (Exception, GithubException) as e:
        logging.error(f"Error fetching file content for {file_path}: {e}")
        return None
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

# Function to process a pull request
def process_pull_request(pr):
    try:
        # Lists to store issues
        pr_complexity_scores = []
        pr_security_issues = []
        pr_code_quality_issues = []

        for file in pr.get_files():
            file_content = get_file_content(repo, file.filename)
            if file_content:
                language = detect_language(file.filename)
                if language:
                    # Run security checks and capture any issues
                    security_issues = run_security_checks(file_content, language)
                    if security_issues:
                        pr_security_issues.append(security_issues)
                    
                    # Analyze code quality
                    code_quality_issues = analyze_code_quality(file_content, language)
                    if code_quality_issues:
                        pr_code_quality_issues.append("\n".join(code_quality_issues))
                
                    # Add other analysis tools here...

        # Calculate average complexity score
        avg_complexity_score = sum(pr_complexity_scores) / len(pr_complexity_scores) if pr_complexity_scores else 0

        # Collect PR details
        pr_details = {
            "pr_id": pr.number,
            "repository_name": repo.name,
            "title_length": len(pr.title),
            "description_length": len(pr.body) if pr.body else 0,
            "files_changed": pr.changed_files,
            "lines_added": pr.additions,
            "lines_deleted": pr.deletions,
            "complexity_score": avg_complexity_score,
            "time_to_merge": (pr.merged_at - pr.created_at).total_seconds() / 3600 if pr.merged_at else None,
            "num_commits": pr.commits,
            "is_bug_fix": 'bug' in [label.name for label in pr.get_labels()],
            "num_comments": pr.comments + pr.review_comments,
            "num_reviewers": len(set([review.user.login for review in pr.get_reviews()])),
            "merge_status": pr.merged,
            "author_experience": repo.get_commits(author=pr.user.login).totalCount,
            "author_tenure": (pr.created_at - pr.user.created_at).days / 30,
            "author_contributions": pr.user.contributions,
            "review_time": sum([(review.submitted_at - pr.created_at).total_seconds() / 3600 for review in pr.get_reviews() if review.submitted_at]),
            "num_approvals": len([review for review in pr.get_reviews() if review.state == 'APPROVED']),
            "review_sentiment_score": sum([calculate_sentiment(comment.body) for comment in pr.get_issue_comments()]),
            "security_issues": "\n".join(pr_security_issues),  # Concatenate all security issues found
            "code_quality_issues": "\n".join(pr_code_quality_issues)  # Concatenate all code quality issues found
        }
        logging.info(f"Pull Request Details: {pr_details}")
        return pr_details

    except Exception as e:
        logging.error(f"Error processing PR {pr.number} in {repo.name}: {e}")
        return None

def check_if_csv_exists(file_path):
    if os.path.isfile(file_path):
        print(f"The file '{file_path}' already exists.")
        return True
    else:
        print(f"The file '{file_path}' does not exist.")
        return False

# Example usage
 
def append_to_csv(new_data, file_path):
    # Check if the file already exists
    if os.path.isfile(file_path):
        # Load the existing CSV file
        existing_df = pd.read_csv(file_path)
        # Create a DataFrame from the new data
        new_df = pd.DataFrame(new_data)
        # Append the new data to the existing DataFrame
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
        # Save the updated DataFrame to the CSV file
        updated_df.to_csv(file_path, index=False)
    else:
        # If the file doesn't exist, create it and save the new data
        pd.DataFrame(new_data).to_csv(file_path, index=False)

# Initialize a list to hold PR data
# pr_data = []

# # Loop through repositories
# for repo_name in repositories:
#     repo = fetch_with_retry(repo_name)
#     if not repo:
#         continue
#     try:
#         repo = g.get_repo(repo_name)
#         print("Repository_Name: ", repo_name)
#         print("Rate_Limit: ",rate_limit)
#         logging.info(f"Processing repo is: {repo_name}")
#     except Exception as e:
#         print(f"Error accessing repository {repo_name}: {e}")
#         continue  # Skip to the next repository if there's an error

#     # Loop through pull requests in the repository
#     try:
#         pull_requests = repo.get_pulls(state='all')  # Get all PRs (open, closed, merged)

#         # Use ThreadPoolExecutor to process PRs concurrently
#         with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
#             future_to_pr = {executor.submit(process_pull_request, pr): pr for pr in pull_requests}
#             for future in concurrent.futures.as_completed(future_to_pr):
#                 pr_details = future.result()
#                 if pr_details:
#                     pr_data.append(pr_details)

#         # Calculate repository-wide features
#         repo_activity_level = len(list(repo.get_commits()))
#         repo_size = repo.size / 1024  # Convert repository size from KB to MB

#         # Add repository-wide features to each PR entry
#         for pr in pr_data:
#             pr['repo_activity_level'] = repo_activity_level
#             pr['repo_size'] = repo_size

#     except Exception as e:
#         print(f"Error processing repository {repo_name}: {e}")
#         continue  # Skip to the next repository if there's an error

# # Convert the list of dictionaries to a pandas DataFrame
# print("before save pr_data: ", pr_data)
# if pr_data:
#     df = pd.DataFrame(pr_data)
#     print("DataFrame created successfully.")
#     print(df.head())  # Check the first few rows of the DataFrame

#     # Export the DataFrame to a CSV file
    
#     output_file = 'new_dataset.csv'
#     try:
#         if check_if_csv_exists(output_file):
#             append_to_csv(pr_data,output_file)
#             print(f"Dataset updated to {output_file}")
#         else:
#             df.to_csv(output_file, index=False)
#             print(f"Dataset saved to {output_file}")
#     except Exception as e:
#         print(f"Failed to save dataset: {e}")
# else:
#     print("No PR data to save.")
pr_data = []

# Loop through repositories
for repo_name in repositories:
    # Fetch repository data, trying to load from cache first
    repo = fetch_with_retry(repo_name)
    print("repo_type: ",type(repo))
    if repo is None:
        logging.warning(f"Skipping repository {repo_name} due to access issues or missing data.")
        continue
    
    # Loop through pull requests in the repository
    for pr in repo.get_pulls(state='all'):
        try:
            # Process each pull request
            pr_details = process_pull_request(pr)
            
            if pr_details:
                pr_data.append(pr_details)
                
                # Save the data to CSV after processing each PR
                csv_file_path = f"{repo_name.replace('/', '_')}_prs.csv"
                append_to_csv([pr_details], csv_file_path)
                
                # Log successful processing
                logging.info(f"Processed PR {pr.number} in {repo_name} and saved to CSV.")
            
        except Exception as e:
            logging.error(f"Error processing PR {pr.number} in {repo_name}: {e}")
            continue
