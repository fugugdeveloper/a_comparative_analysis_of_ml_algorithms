import base64
import json
import mimetypes
import os
import re
import sys
import tempfile
import time
from github import Github, GithubException
import pandas as pd
import requests
from textblob import TextBlob
import concurrent.futures
import logging
import subprocess
import radon.complexity as radon_complexity

logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(encoding='utf-8')

ACCESS_TOKENS = [
    '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',#for codeigniter
    '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',#for react
    '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',# for java
    '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',# for dango
    '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',#for rails
    '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',#for cpp
'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', #free


    # Add more tokens as needed
]
languages_ext={
    'Python': '.py',
    'Java': '.java',
    'JavaScript': '.js',
    'C++': '.cpp',
    'C': '.c',
    'Go': '.go',
    'PHP': '.php',
    'Ruby': '.rb',
    'C#': '.cs',
    'TypeScript': '.ts',
    'HTML': '.html',
    'CSS': '.css',
    'XML': '.xml',
    'JSON': '.json',
    'Shell': '.sh',
    'Batch': '.bat',
    'Rust': '.rs',
    'Swift': '.swift',
    'Kotlin': '.kt',
}

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
}
# Authenticate to GitHub
current_token_index = 0
g = Github(ACCESS_TOKENS[current_token_index], timeout=120)
rate_limit = g.get_rate_limit()
print("Rate_Limit: ", rate_limit)
# List of repositories you want to analyze
CACHE_DIR = "github_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
repositories = [
    "codeigniter4/CodeIgniter4",
    "facebook/react",
    "spring-projects/spring-framework",
    "django/django",
    "microsoft/vscode",
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
def get_file_extension(language):
    return languages_ext.get(language, "Unknown language")

def detect_language(file_name):
    ext = file_name.split('.')[-1]
    if ext in ['py']:
        return 'Python'
    elif ext in ['js', 'ts']:
        return 'JavaScript'
    elif ext in ['java']:
        return 'Java'
    elif ext in ['cpp']:
        return 'C++'
    elif ext in ['c']:
        return  'C'
    elif ext in ['go']:
        return 'GO'
    elif ext in ['rb']:
        return 'Ruby'
    elif ext in ['php']:
        return 'PHP'
    elif ext in ['cs']:
        return 'C#'
    elif ext in ['ts']:
        return 'TypeScript'
    elif ext in ['html']:
        return 'HTML'
    elif ext in ['json']:
        return 'JSON'
    elif ext in ['sh']:
        return 'Shell'
    elif ext in ['bat']:
        return 'Batch'
    elif ext in ['rs']:
        return 'Rust'
    elif ext in ['swift']:
        return 'Swift'
    elif ext in ['kt']:
        return 'Kotlin'
    elif ext in ['xml']:
        return 'XML'
    else:
        return None

def language_extension(filename):
    ext = filename.split('.')[-1]
    if ext in ['py']:
        return '.py'
    elif ext in ['js']:
        return '.js'
    elif ext in ['ts']:
        return '.ts'
    elif ext in ['java']:
        return '.java'
    elif ext in ['cpp', 'h']:
        return '.cpp'
    elif ext in ['go']:
        return '.go'
    elif ext in ['rb']:
        return '.rb'
    else:
        return None

# Function to run security checks based on language
def run_security_checks(file_content, language):
    if language == 'Python':
        return run_bandit_security_checks(file_content)
    elif language == 'JavaScript':
        return run_eslint_security_checks(file_content)
    elif language == 'Java':
        return run_spotbugs_security_checks(file_content)
    elif language == 'C++':
        return run_cppcheck_security_checks(file_content)
    elif language == 'GO':
        return run_gosec_security_checks(file_content)
    elif language == 'Ruby':
        return run_brakeman_security_checks(file_content)
    else:
        return None


# Function to run Bandit for Python

def run_bandit_security_checks(file_content):
    try:
        if isinstance(file_content, tuple):
            file_content = ''.join(file_content)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_filename = temp_file.name
            temp_file.write(file_content)
            temp_file.flush()
            # Ensure all data is written to the file
            # Run Bandit on the temporary file
        result = subprocess.run(['bandit', '-r', temp_filename], capture_output=True, text=True)
        print("Bandit Result: ", result.stdout)
        return result.stdout
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
    try:
        if isinstance(file_content, tuple):
            file_content = ''.join(file_content)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as temp_file:
            temp_filename = temp_file.name
            temp_file.write(file_content)
            temp_file.flush()  # Ensure all data is written to the file

        # Run Elsint on the temporary file
        result = subprocess.run(['C:/Users/user/AppData/Roaming/npm/eslint.cmd', temp_filename], capture_output=True,
                                text=True)
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
    try:
        if isinstance(file_content, tuple):
            file_content = ''.join(file_content)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.java', delete=False) as temp_file:
            temp_filename = temp_file.name
            temp_file.write(file_content)
            temp_file.flush()  # Ensure all data is written to the file

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
    try:
        if isinstance(file_content, tuple):
            file_content = ''.join(file_content)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as temp_file:
            temp_filename = temp_file.name
            temp_file.write(file_content)
            temp_file.flush()  # Ensure all data is written to the file

        # Run Cppcheck on the temporary file
        result = subprocess.run(['C:/Program Files/Cppcheck/cppcheck.exe', '--enable=all', temp_filename],
                                capture_output=True, text=True)
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

    try:
        if isinstance(file_content, tuple):
            file_content = ''.join(file_content)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.go', delete=False) as temp_file:
            temp_filename = temp_file.name
            temp_file.write(file_content)
            temp_file.flush()  # Ensure all data is written to the file
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
    try:
        if isinstance(file_content, tuple):
            file_content = ''.join(file_content)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.rb', delete=False) as temp_file:
            temp_filename = temp_file.name
            temp_file.write(file_content)
            temp_file.flush()  # Ensure all data is written to the file
        # Run Brakeman on the temporary file
        result = subprocess.run(['C:\Ruby32-x64\bin\brakeman.bat', temp_filename], capture_output=True, text=True)
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


def extract_lizard_metrics(file_content, language_ext):
    try:
        python_executable = sys.executable
        if isinstance(file_content, tuple):
            file_content = ''.join(file_content)
        if language_ext !='Unknown language':
            with tempfile.NamedTemporaryFile(mode='w', suffix=language_ext, delete=False) as temp_file:
                temp_filename = temp_file.name
                temp_file.write(file_content)
                temp_file.flush()
            result = subprocess.run([python_executable, '-m', 'lizard', temp_filename], capture_output=True, text=True)
            print("result lizard: ", result.stdout)

        # Initialize metrics dictionary
            metrics = {
                'NLOC': None,
                'Avg.NLOC': None,
                'AvgCCN': None,
                'Avg.token': None,
                'Fun Cnt': None,
                'Warning cnt': None,
                'Fun Rt': None,
                'nloc Rt': None
            }

            # Parse the Lizard output
            lines = result.stdout.splitlines()
            metrics_section_started = False

            for line in lines:
                if line.startswith("Total"):
                    metrics_section_started = True
                    continue
                if metrics_section_started:
                    parts = line.split()
                    if len(parts) == 8:  # Adjust based on the actual number of columns
                        metrics['NLOC'] = parts[0]
                        metrics['Avg.NLOC'] = parts[1]
                        metrics['AvgCCN'] = parts[2]
                        metrics['Avg.token'] = parts[3]
                        metrics['Fun Cnt'] = parts[4]
                        metrics['Warning cnt'] = parts[5]
                        metrics['Fun Rt'] = parts[6]
                        metrics['nloc Rt'] = parts[7]
                        break

            # Return metrics in the required order
            return (
                metrics['NLOC'],
                metrics['Avg.NLOC'],
                metrics['AvgCCN'],
                metrics['Avg.token'],
                metrics['Fun Cnt'],
                metrics['Warning cnt'],
                metrics['Fun Rt'],
                metrics['nloc Rt']
            )
        else:
             return None, None, None, None, None, None, None, None

    except Exception as e:
        print(f"Error running Lizard: {e}")
        return None, None, None, None, None, None, None, None
    finally:
        # Ensure the temporary file is deleted after processing
        try:
            os.remove(temp_filename)
        except Exception as e:
            print(f"Error removing temp file {temp_filename}: {e}")


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
    # cached_data = load_from_cache(cache_key)
    #
    # if cached_data:
    #     logging.info(f"Loaded data from cache for repository: {repo_name}")
    #     return cached_data

    for i in range(retries):
        try:
            repo = g.get_repo(repo_name)
            #save_to_cache(repo.raw_data, cache_key)  # Save repo data to cache
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
    # cached_data = load_from_cache(cache_key)
    #
    # if cached_data:
    #     logging.info(f"Loaded file content from cache for file: {file_path}")
    #     return cached_data

    try:
        file_content = process_file(file_path, repo)
        #save_to_cache(file_content, cache_key)  # Save file content to cache
        #print("file_content: ", file_content)
        return file_content
    except (Exception, GithubException) as e:
        logging.error(f"Error fetching file content for {file_path}: {e}")
        return None


# Function to calculate sentiment of review comments
def calculate_sentiment(comment):
    return TextBlob(comment).sentiment.polarity

def analyze_code_quality(file_content, language, precalculated_metrics):
    issues = []
    results = {}

    # Define maximum line lengths for different languages
    line_length_limits = {
        'Python': 79,
        'JavaScript': 80,
        'Java': 100,
        'C++': 100,
        'Go': 80,
        'Ruby': 80,
        'TypeScript': 80,
        'PHP': 80,
        'Swift': 100,
        'Kotlin': 100
    }

    # Define comment patterns for different languages
    comment_patterns = {
        'Python': r'#',
        'JavaScript': r'//|/\*|\*/',
        'Java': r'//|/\*|\*/',
        'C++': r'//|/\*|\*/',
        'Go': r'//|/\*|\*/',
        'Ruby': r'#',
        'TypeScript': r'//|/\*|\*/',
        'PHP': r'//|/\*|\*/',
        'Swift': r'//|/\*|\*/',
        'Kotlin': r'//|/\*|\*/'
    }

    # Define specific pattern checks for each language
    language_specific_patterns = {
        'Python': {
            'print(': "Print statements found."
        },
        'JavaScript': {
            'console.log(': "Console.log statements found."
        },
        'Java': {
            'System.out.println(': "System.out.println statements found."
        },
        'C++': {
            '#include': "Header includes found."
        },
        'Go': {
            'fmt.Println(': "fmt.Println statements found."
        },
        'Ruby': {
            'puts ': "Put statements found."
        },
        'TypeScript': {
            'console.log(': "Console.log statements found."
        },
        'PHP': {
            'echo ': "Echo statements found."
        },
        'Swift': {
            'print(': "Print statements found."
        },
        'Kotlin': {
            'println(': "Println statements found."
        }
    }

    # Check for language-specific issues
    for pattern, message in language_specific_patterns.get(language, {}).items():
        if pattern in file_content:
            issues.append(message)

    # Check for commented-out code
    comment_pattern = comment_patterns.get(language, r'#')
    if re.search(comment_pattern + r'.*\b(if|for|while|return|class|def|function)\b', file_content):
        issues.append("Commented-out code found.")

    # Check for long lines
    max_line_length = line_length_limits.get(language, 80)
    long_lines = [line for line in file_content.splitlines() if len(line) > max_line_length]
    if long_lines:
        issues.append(f"{len(long_lines)} lines exceed {max_line_length} characters.")

    # Use precalculated metrics
    metrics = precalculated_metrics

    # Append precalculated metrics to results
    results['NLOC'] = metrics.get('NLOC', 'Not provided')
    results['Avg_NLOC'] = metrics.get('Avg_NLOC', 'Not provided')
    results['AvgCCN'] = metrics.get('AvgCCN', 'Not provided')
    results['Avg_token'] = metrics.get('Avg_token', 'Not provided')
    results['Fun_Cnt'] = metrics.get('Fun_Cnt', 'Not provided')
    results['Warning_cnt'] = metrics.get('Warning_cnt', 'Not provided')
    results['Fun_Rt'] = metrics.get('Fun_Rt', 'Not provided')
    results['nloc_Rt'] = metrics.get('nloc_Rt', 'Not provided')

    # Classify overall quality based on metrics
    quality_rating = classify_quality(metrics)

    # Add quality rating to results
    results['Quality Rating'] = quality_rating

    # Summary of the issues
    issues_summary = issues if issues else ["No major code quality issues found."]

    return results, issues_summary

# code quality rate
def classify_quality(metrics):
    # Define thresholds and margin of flexibility
    thresholds = {
        'worst': {'NLOC': 1000000, 'Avg_NLOC': 150, 'AvgCCN': 15, 'Avg_token': 1000, 'Fun_Cnt': 400, 'Warning_cnt': 1,
                  'Fun_Rt': 1.5, 'nloc_Rt': 1.0},
        'bad': {'NLOC': 500000, 'Avg_NLOC': 100, 'AvgCCN': 15, 'Avg_token': 700, 'Fun_Cnt': 300, 'Warning_cnt': 1,
                'Fun_Rt': 2.0, 'nloc_Rt': 1.5},
        'not bad': {'NLOC': 300000, 'Avg_NLOC': 75, 'AvgCCN': 9.5, 'Avg_token': 500, 'Fun_Cnt': 200, 'Warning_cnt': 11,
                    'Fun_Rt': 1.5, 'nloc_Rt': 1.2},
        'medium': {'NLOC': 200000, 'Avg_NLOC': 50, 'AvgCCN': 6.5, 'Avg_token': 300, 'Fun_Cnt': 100, 'Warning_cnt': 0,
                   'Fun_Rt': 1.2, 'nloc_Rt': 1.0},
        'good': {'NLOC': 10000, 'Avg_NLOC': 40, 'AvgCCN': 4.5, 'Avg_token': 150, 'Fun_Cnt': 50, 'Warning_cnt': 0,
                 'Fun_Rt': 1.0, 'nloc_Rt': 0.8},
        'great': {'NLOC': 5000, 'Avg_NLOC': 30, 'AvgCCN': 2.5, 'Avg_token': 70, 'Fun_Cnt': 30, 'Warning_cnt': 0,
                  'Fun_Rt': 0.8, 'nloc_Rt': 0.6},
        'best': {'NLOC': 2000, 'Avg_NLOC': 20, 'AvgCCN': 1.5, 'Avg_token': 50, 'Fun_Cnt': 20, 'Warning_cnt': 0, 'Fun_Rt': 0.6,
                 'nloc_Rt': 0},
        'the best': {'NLOC': 1000, 'Avg_NLOC': 10, 'AvgCCN': 1, 'Avg_token': 30, 'Fun_Cnt': 10, 'Warning_cnt': 0,
                     'Fun_Rt': 0, 'nloc_Rt': 0},
    }
    margin = 0.1  # Margin for flexibility (10%)

    # Extract metrics and convert them to numeric types
    nloc = float(metrics.get('NLOC', 0))
    avg_nloc = float(metrics.get('Avg_NLOC', 0))
    avg_ccn = float(metrics.get('AvgCCN', 0))
    avg_token = float(metrics.get('Avg_token', 0))
    fun_cnt = float(metrics.get('Fun_Cnt', 0))
    warning_cnt = float(metrics.get('Warning_cnt', 0))
    fun_rt = float(metrics.get('Fun_Rt', 0))
    nloc_rt = float(metrics.get('nloc_Rt', 0))

    # Calculate scores based on thresholds and margins
    scores = {}
    for rating, threshold in sorted(thresholds.items(), key=lambda x: x[1]['NLOC']):
        score = 0
        for key in threshold.keys():
            metric_value = locals().get(key.lower())
            threshold_value = threshold[key]
            if metric_value is None:
                continue
            if metric_value <= threshold_value * (1 + margin):
                score += 1
            elif metric_value <= threshold_value:
                score += 0.5
        scores[rating] = score / len(threshold)  # Normalize score

    # Determine quality based on scores
    best_rating = max(scores, key=scores.get)
    return best_rating



# Function to process a pull request
def process_pull_request(pr):
    try:
        # Lists to store issues
        pr_complexity_scores = []
        pr_security_issues = []
        pr_code_quality_issues = None
        code_quality_rates=None
        for file in pr.get_files():
            file_content = get_file_content(repo, file.filename)
            if file_content:
                language = detect_language(file.filename)
                language_ext=get_file_extension(language)
                if language:
                    # Run security checks and capture any issues
                    security_issues = run_security_checks(file_content, language)
                    if security_issues:
                        pr_security_issues.append(security_issues)
                if language_ext:
                    NLOC, Avg_NLOC, AvgCCN, Avg_token, Fun_Cnt, Warning_cnt, Fun_Rt, nloc_Rt = extract_lizard_metrics(file_content, language_ext)
                print("NLOC: ", NLOC)
                print("Avg.NLOC: ", Avg_NLOC)
                print("AvgCCN: ", AvgCCN)
                print("Avg.token: ", Avg_token)
                print("Fun Cnt: ", Fun_Cnt)
                print("Warning cnt: ", Warning_cnt)
                print("Fun Rt: ", Fun_Rt)
                print("nloc Rt: ", nloc_Rt)

                    # Analyze code quality
                precalculated_metrics = {
                    'NLOC': NLOC,
                    'Avg_NLOC': Avg_NLOC,
                    'AvgCCN': AvgCCN,
                    'Avg_token': Avg_token,
                    'Fun_Cnt': Fun_Cnt,
                    'Warning_cnt': Warning_cnt,
                    'Fun_Rt': Fun_Rt,
                    'nloc_Rt': nloc_Rt
                }

                # Analyze the code quality
                results, issues_summary = analyze_code_quality(file.filename, language, precalculated_metrics)

                # Print numerical results
                print("Numerical Results:")
                for key, value in results.items():
                    print(f"Key: {key}: Value: {value}")
                    if key == 'Quality Rating':
                        code_quality_rates=value
                        print(f"Key: {key}: Value: {value}")

                # Print issues summary

                for issue in issues_summary:
                    print("\nIssues Summary: ", issue)
                    pr_code_quality_issues=issue
                    #pr_code_quality_issues.append(issue)
                # code_quality_issues = analyze_code_quality(file_content, language)
                # if code_quality_issues:
                #         pr_code_quality_issues.append("\n".join(code_quality_issues))

                # Add other analysis tools here...
        pr_details = {
            "pr_id": pr.number,
            "repository_name": repo.name,
            "title_length": len(pr.title),
            "description_length": len(pr.body) if pr.body else 0,
            "files_changed": pr.changed_files,
            "lines_added": pr.additions,
            "lines_deleted": pr.deletions,
            "time_to_merge": (pr.merged_at - pr.created_at).total_seconds() / 3600 if pr.merged_at else None,
            "num_commits": pr.commits,
            "is_bug_fix": 'bug' in [label.name for label in pr.get_labels()],
            "num_comments": pr.comments + pr.review_comments,
            "num_reviewers": len(set([review.user.login for review in pr.get_reviews()])),
            "merge_status": pr.merged,
            "author_experience": repo.get_commits(author=pr.user.login).totalCount,
            "author_tenure": (pr.created_at - pr.user.created_at).days / 30,
            "author_contributions": pr.user.contributions,
            "review_time": sum(
                [(review.submitted_at - pr.created_at).total_seconds() / 3600 for review in pr.get_reviews() if
                 review.submitted_at]),
            "num_approvals": len([review for review in pr.get_reviews() if review.state == 'APPROVED']),
            "review_sentiment_score": sum([calculate_sentiment(comment.body) for comment in pr.get_issue_comments()]),
            "security_issues": "\n".join(pr_security_issues),  # Concatenate all security issues found
            "code_quality_issues": "\n".join(pr_code_quality_issues),
            "code_quality_rate": code_quality_rates,


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


def append_to_csv(data, csv_path):
    df = pd.DataFrame(data)
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)

pr_data = []

# Loop through repositories
for repo_name in repositories:
    # Fetch repository data, trying to load from cache first
    repo = fetch_with_retry(repo_name)
    print("repo_type: ", type(repo))
    if repo is None:
        logging.warning(f"Skipping repository {repo_name} due to access issues or missing data.")
        continue

    # Loop through pull requests in the repository
    for pr in repo.get_pulls(state='closed'):
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
