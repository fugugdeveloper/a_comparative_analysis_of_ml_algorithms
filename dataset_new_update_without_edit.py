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

# Set up your GitHub personal access tokens
ACCESS_TOKENS = [
    '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',
    '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',
    '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',
    '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',
    # Add more tokens as needed
]

# Authenticate to GitHub
current_token_index = 0
g = Github(ACCESS_TOKENS[current_token_index], timeout=120)
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

def handle_rate_limit():
    rate_limit = g.get_rate_limit()
    if rate_limit.core.remaining == 0:
        logging.info("Rate limit exceeded. Switching to the next token.")
        switch_token()  # Switch to the next token

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

def run_bandit_security_checks(file_content):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
        temp_filename = temp_file.name
        temp_file.write(file_content)
        temp_file.flush()

    try:
        result = subprocess.run(['bandit', '-r', temp_filename], capture_output=True, text=True)
        print("Bandit Result: ", result.stdout)
    except Exception as e:
        print(f"Error running Bandit: {e}")
    finally:
        try:
            os.remove(temp_filename)
        except Exception as e:
            print(f"Error removing temp file {temp_filename}: {e}")
    
    return result.stdout if 'result' in locals() else None

def run_eslint_security_checks(file_content):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as temp_file:
        temp_filename = temp_file.name
        temp_file.write(file_content)
        temp_file.flush()

    try:
        result = subprocess.run(['C:/Users/user/AppData/Roaming/npm/eslint.cmd', temp_filename], capture_output=True, text=True)
        print("Eslint Result: ", result.stdout)
    except Exception as e:
        print(f"Error running Eslint: {e}")
    finally:
        try:
            os.remove(temp_filename)
        except Exception as e:
            print(f"Error removing temp file {temp_filename}: {e}")
    
    return result.stdout if 'result' in locals() else None

def run_spotbugs_security_checks(file_content):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.java', delete=False) as temp_file:
        temp_filename = temp_file.name
        temp_file.write(file_content)
        temp_file.flush()

    try:
        result = subprocess.run(
            ['C:/spotbugs/bin/spotbugs.bat', '-textui', '-high', '-effort:max', temp_filename],
            capture_output=True, text=True
        )
        print("SpotBugs Result: ", result.stdout)
    except Exception as e:
        print(f"Error running SpotBugs: {e}")
    finally:
        try:
            os.remove(temp_filename)
        except Exception as e:
            print(f"Error removing temp file {temp_filename}: {e}")
    
    return result.stdout if 'result' in locals() else None

def run_cppcheck_security_checks(file_content):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as temp_file:
        temp_filename = temp_file.name
        temp_file.write(file_content)
        temp_file.flush()

    try:
        result = subprocess.run(['C:/Program Files/Cppcheck/cppcheck.exe', '--enable=all', temp_filename], capture_output=True, text=True)
        print("Cppcheck Result: ", result.stdout)
    except Exception as e:
        print(f"Error running Cppcheck: {e}")
    finally:
        try:
            os.remove(temp_filename)
        except Exception as e:
            print(f"Error removing temp file {temp_filename}: {e}")
    
    return result.stdout if 'result' in locals() else None

def run_gosec_security_checks(file_content):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.go', delete=False) as temp_file:
        temp_filename = temp_file.name
        temp_file.write(file_content)
        temp_file.flush()

    try:
        result = subprocess.run(['gosec', temp_filename], capture_output=True, text=True)
        print("Gosec Result: ", result.stdout)
    except Exception as e:
        print(f"Error running Gosec: {e}")
    finally:
        try:
            os.remove(temp_filename)
        except Exception as e:
            print(f"Error removing temp file {temp_filename}: {e}")
    
    return result.stdout if 'result' in locals() else None

def run_brakeman_security_checks(file_content):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.rb', delete=False) as temp_file:
        temp_filename = temp_file.name
        temp_file.write(file_content)
        temp_file.flush()

    try:
        result = subprocess.run(['brakeman', temp_filename], capture_output=True, text=True)
        print("Brakeman Result: ", result.stdout)
    except Exception as e:
        print(f"Error running Brakeman: {e}")
    finally:
        try:
            os.remove(temp_filename)
        except Exception as e:
            print(f"Error removing temp file {temp_filename}: {e}")
    
    return result.stdout if 'result' in locals() else None

def generate_cache_key(repo_name, pr_number=None):
    key = f"{repo_name}_pr_{pr_number}.json" if pr_number else f"{repo_name}.json"
    return os.path.join(CACHE_DIR, key)

def save_to_cache(data, cache_key):
    os.makedirs(os.path.dirname(cache_key), exist_ok=True)
    json_data = json.dumps(data, indent=4)
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
                    logging.warning(f"Request timed out. Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    logging.error(f"Request timed out after {retries} retries.")
                    return None
            else:
                raise  # Re-raise exception if it's not related to rate limiting
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed: {e}")
            return None
    return None

def fetch_prs_with_retry(repo, retries=5, backoff=2):
    cache_key = generate_cache_key(repo.full_name, "prs")
    cached_data = load_from_cache(cache_key)

    if cached_data:
        logging.info(f"Loaded PR data from cache for repository: {repo.full_name}")
        return cached_data

    for i in range(retries):
        try:
            prs = repo.get_pulls(state='all')
            prs_data = [pr.raw_data for pr in prs]
            save_to_cache(prs_data, cache_key)  # Save PR data to cache
            return prs
        except GithubException as e:
            if e.status == 403 and 'rate limit' in str(e):
                handle_rate_limit()
                continue  # Retry after handling rate limit
            elif e.status == 403:
                logging.warning(f"Access forbidden to PRs for repository: {repo.full_name}. Check your token and permissions.")
                return None
            elif e.status == 404:
                logging.warning(f"PRs not found for repository: {repo.full_name}.")
                return None
            elif isinstance(e, requests.exceptions.ReadTimeout):
                if i < retries - 1:
                    sleep_time = backoff * (2 ** i)
                    logging.warning(f"Request timed out. Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    logging.error(f"Request timed out after {retries} retries.")
                    return None
            else:
                raise  # Re-raise exception if it's not related to rate limiting
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed: {e}")
            return None
    return None

def analyze_repository(repo_name):
    logging.info(f"Analyzing repository: {repo_name}")
    repo = fetch_with_retry(repo_name)
    if not repo:
        logging.error(f"Failed to fetch repository: {repo_name}")
        return
    
    prs = fetch_prs_with_retry(repo)
    if prs is None:
        logging.error(f"Failed to fetch PRs for repository: {repo_name}")
        return

    results = []
    for pr in prs:
        logging.info(f"Processing PR #{pr.number} from repository: {repo_name}")
        files = pr.get_files()
        for file in files:
            language = detect_language(file.filename)
            if language:
                result = run_security_checks(file.patch, language)
                results.append({
                    "repo_name": repo_name,
                    "pr_number": pr.number,
                    "file": file.filename,
                    "language": language,
                    "security_analysis": result
                })
    
    return results

def main():
    all_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_repo = {executor.submit(analyze_repository, repo): repo for repo in repositories}
        for future in concurrent.futures.as_completed(future_to_repo):
            repo = future_to_repo[future]
            try:
                repo_results = future.result()
                if repo_results:
                    all_results.extend(repo_results)
            except Exception as exc:
                logging.error(f"Repository {repo} generated an exception: {exc}")

    # Convert the results to a DataFrame
    df = pd.DataFrame(all_results)
    df.to_csv('repo_analysis_results.csv', index=False)
    logging.info("Analysis complete. Results saved to repo_analysis_results.csv")

if __name__ == "__main__":
    main()
