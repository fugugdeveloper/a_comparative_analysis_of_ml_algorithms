import json
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

# Set up your GitHub personal access token
ACCESS_TOKEN = os.getenv('GITHUB_TOKEN', 'your_github_token_here')

# Authenticate to GitHub
g = Github(ACCESS_TOKEN, timeout=120)
rate_limit = g.get_rate_limit()
print("Rate_Limit: ", rate_limit)
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

def handle_rate_limit():
    rate_limit = g.get_rate_limit()
    if rate_limit.rate.remaining == 0:
        reset_time = rate_limit.rate.reset
        wait_time = (reset_time - time.time()).total_seconds()
        logging.info(f"Rate limit exceeded. Waiting for {wait_time} seconds.")
        time.sleep(wait_time + 10)

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

# Bandit, ESLint, SpotBugs, Cppcheck, Gosec, and Brakeman functions omitted for brevity

def generate_cache_key(repo_name, file_path=None):
    key = f"{repo_name}_{file_path.replace('/', '_')}.json" if file_path else f"{repo_name}.json"
    return os.path.join(CACHE_DIR, key)

def save_to_cache(data, cache_key):
    os.makedirs(os.path.dirname(cache_key), exist_ok=True)
    with open(cache_key, 'w') as cache_file:
        json.dump(data, cache_file)

def load_from_cache(cache_key):
    if os.path.exists(cache_key):
        with open(cache_key, 'r') as cache_file:
            return json.load(cache_file)
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
            if e.status == 403:
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

def get_file_content(repo, file_path):
    cache_key = generate_cache_key(repo.full_name, file_path)
    cached_data = load_from_cache(cache_key)
    
    if cached_data:
        logging.info(f"Loaded file content from cache for file: {file_path}")
        return cached_data

    try:
        file_content = repo.get_contents(file_path).decoded_content.decode('utf-8')
        save_to_cache(file_content, cache_key)  # Save file content to cache
        return file_content
    except GithubException as e:
        if e.status == 403:
            logging.warning(f"Access forbidden for file: {file_path}")
        elif e.status == 404:
            logging.warning(f"File not found: {file_path}")
        elif e.status == 403 and 'rate limit' in str(e):
            handle_rate_limit()
            return get_file_content(repo, file_path)  # Retry after handling rate limit
        else:
            logging.error(f"Error fetching file content for {file_path}: {e}")
        return None

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

def process_pull_request(pr):
    try:
        pr_complexity_scores = []
        pr_security_issues = []
        pr_code_quality_issues = []

        for file in pr.get_files():
            file_content = get_file_content(repo, file.filename)
            if file_content:
                language = detect_language(file.filename)
                if language:
                    security_issues = run_security_checks(file_content, language)
                    if security_issues:
                        pr_security_issues.append(security_issues)
                    
                    code_quality_issues = analyze_code_quality(file_content, language)
                    if code_quality_issues:
                        pr_code_quality_issues.append("\n".join(code_quality_issues))
                
        avg_complexity_score = sum(pr_complexity_scores) / len(pr_complexity_scores) if pr_complexity_scores else 0

        pr_details = {
            "pr_id": pr.number,
            "repository_name": repo.name,
            "title_length": len(pr.title),
            "description_length": len(pr.body) if pr.body else 0,
            "files_changed": pr.changed_files,
            "lines_added": pr.additions,
            "lines_deleted": pr.deletions,
            "complexity_score": avg_complexity_score,
            "time_to_merge": (pr.merged_at - pr.created
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
            if e.status == 403:
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
