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
import radon.complexity as radon_complexity
import logging
import subprocess
from urllib3.exceptions import NewConnectionError, MaxRetryError

logging.basicConfig(level=logging.DEBUG)

# Set up your GitHub personal access tokens
ACCESS_TOKENS = [
    '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',
    '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',
    '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',
    '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',
    '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',
    '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'
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
    "opencv/opencv",
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
    for _ in range(len(ACCESS_TOKENS)):
        rate_limit = g.get_rate_limit()
        if rate_limit.core.remaining > 0:
            return  # Found a token with remaining rate limit
        else:
            logging.info(
                f"Token {current_token_index + 1} is exhausted. Switching to the next token."
            )
            switch_token()
    logging.error("All tokens have been exhausted.")
    raise Exception("Rate limit reached on all tokens.")


def detect_language(file_name):
    ext = file_name.split(".")[-1]
    if ext in ["py"]:
        return "python"
    elif ext in ["js", "ts"]:
        return "javascript"
    elif ext in ["java"]:
        return "java"
    elif ext in ["c", "cpp", "h"]:
        return "cpp"
    elif ext in ["go"]:
        return "go"
    elif ext in ["rb"]:
        return "ruby"
    else:
        return None


def run_security_checks(file_content, language):
    if language == "python":
        return run_bandit_security_checks(file_content)
    elif language == "javascript":
        return run_eslint_security_checks(file_content)
    elif language == "java":
        return run_spotbugs_security_checks(file_content)
    elif language == "cpp":
        return run_cppcheck_security_checks(file_content)
    elif language == "go":
        return run_gosec_security_checks(file_content)
    elif language == "ruby":
        return run_brakeman_security_checks(file_content)
    else:
        return None


def run_bandit_security_checks(file_content):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as temp_file:
        temp_filename = temp_file.name
        temp_file.write(file_content)
        temp_file.flush()

    try:
        result = subprocess.run(
            ["bandit", "-r", temp_filename], capture_output=True, text=True
        )
        print("Bandit Result: ", result.stdout)
    except Exception as e:
        print(f"Error running Bandit: {e}")
    finally:
        try:
            os.remove(temp_filename)
        except Exception as e:
            print(f"Error removing temp file {temp_filename}: {e}")

    return result.stdout if "result" in locals() else None


def run_eslint_security_checks(file_content):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as temp_file:
        temp_filename = temp_file.name
        temp_file.write(file_content)
        temp_file.flush()

    try:
        result = subprocess.run(
            ["C:/Users/user/AppData/Roaming/npm/eslint.cmd", temp_filename],
            capture_output=True,
            text=True,
        )
        print("Eslint Result: ", result.stdout)
    except Exception as e:
        print(f"Error running Eslint: {e}")
    finally:
        try:
            os.remove(temp_filename)
        except Exception as e:
            print(f"Error removing temp file {temp_filename}: {e}")

    return result.stdout if "result" in locals() else None


def run_spotbugs_security_checks(file_content):
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".java", delete=False
    ) as temp_file:
        temp_filename = temp_file.name
        temp_file.write(file_content)
        temp_file.flush()

    try:
        result = subprocess.run(
            [
                "C:/spotbugs/bin/spotbugs.bat",
                "-textui",
                "-high",
                "-effort:max",
                temp_filename,
            ],
            capture_output=True,
            text=True,
        )
        print("SpotBugs Result: ", result.stdout)
    except Exception as e:
        print(f"Error running SpotBugs: {e}")
    finally:
        try:
            os.remove(temp_filename)
        except Exception as e:
            print(f"Error removing temp file {temp_filename}: {e}")

    return result.stdout if "result" in locals() else None


def run_cppcheck_security_checks(file_content):
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".cpp", delete=False
    ) as temp_file:
        temp_filename = temp_file.name
        temp_file.write(file_content)
        temp_file.flush()

    try:
        result = subprocess.run(
            ["C:/Program Files/Cppcheck/cppcheck.exe", "--enable=all", temp_filename],
            capture_output=True,
            text=True,
        )
        print("Cppcheck Result: ", result.stdout)
    except Exception as e:
        print(f"Error running Cppcheck: {e}")
    finally:
        try:
            os.remove(temp_filename)
        except Exception as e:
            print(f"Error removing temp file {temp_filename}: {e}")

    return result.stdout if "result" in locals() else None


def run_gosec_security_checks(file_content):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".go", delete=False) as temp_file:
        temp_filename = temp_file.name
        temp_file.write(file_content)
        temp_file.flush()

    try:
        result = subprocess.run(
            ["gosec", temp_filename], capture_output=True, text=True
        )
        print("Gosec Result: ", result.stdout)
    except Exception as e:
        print(f"Error running Gosec: {e}")
    finally:
        try:
            os.remove(temp_filename)
        except Exception as e:
            print(f"Error removing temp file {temp_filename}: {e}")

    return result.stdout if "result" in locals() else None


def run_brakeman_security_checks(file_content):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".rb", delete=False) as temp_file:
        temp_filename = temp_file.name
        temp_file.write(file_content)
        temp_file.flush()

    try:
        result = subprocess.run(
            ["brakeman", temp_filename], capture_output=True, text=True
        )
        print("Brakeman Result: ", result.stdout)
    except Exception as e:
        print(f"Error running Brakeman: {e}")
    finally:
        try:
            os.remove(temp_filename)
        except Exception as e:
            print(f"Error removing temp file {temp_filename}: {e}")

    return result.stdout if "result" in locals() else None


def analyze_code_quality(file_content, language):
    issues = []
    if language == "python":
        if "print(" in file_content:
            issues.append("Print statements found.")
        if re.search(r"#.*\b(if|for|while|return|class|def)\b", file_content):
            issues.append("Commented-out code found.")
        long_lines = [line for line in file_content.splitlines() if len(line) > 79]
        if long_lines:
            issues.append(f"{len(long_lines)} lines exceed 79 characters.")
    elif language == "javascript":
        if "console.log(" in file_content:
            issues.append("Console.log statements found.")
        if "/*" in file_content or "*/" in file_content:
            issues.append("Commented-out code found.")
        long_lines = [line for line in file_content.splitlines() if len(line) > 80]
        if long_lines:
            issues.append(f"{len(long_lines)} lines exceed 80 characters.")
    elif language == "java":
        if "System.out.println(" in file_content:
            issues.append("System.out.println statements found.")
        if "/*" in file_content or "*/" in file_content:
            issues.append("Commented-out code found.")
        long_lines = [line for line in file_content.splitlines() if len(line) > 100]
        if long_lines:
            issues.append(f"{len(long_lines)} lines exceed 100 characters.")
    elif language == "cpp":
        if "#include" in file_content:
            issues.append("Header includes found.")
        if "/*" in file_content or "*/" in file_content:
            issues.append("Commented-out code found.")
        long_lines = [line for line in file_content.splitlines() if len(line) > 100]
        if long_lines:
            issues.append(f"{len(long_lines)} lines exceed 100 characters.")
    elif language == "go":
        if "fmt.Println(" in file_content:
            issues.append("fmt.Println statements found.")
        if "/*" in file_content or "*/" in file_content:
            issues.append("Commented-out code found.")
        long_lines = [line for line in file_content.splitlines() if len(line) > 80]
        if long_lines:
            issues.append(f"{len(long_lines)} lines exceed 80 characters.")
    elif language == "ruby":
        if "puts " in file_content:
            issues.append("Put statements found.")
        if "# " in file_content:
            issues.append("Commented-out code found.")
        long_lines = [line for line in file_content.splitlines() if len(line) > 80]
        if long_lines:
            issues.append(f"{len(long_lines)} lines exceed 80 characters.")
    return issues if issues else ["No major code quality issues found."]


def calculate_sentiment(comment):
    return TextBlob(comment).sentiment.polarity


def generate_cache_key(repo_name, pr_number=None):
    key = f"{repo_name}_pr_{pr_number}.json" if pr_number else f"{repo_name}.json"
    return os.path.join(CACHE_DIR, key)


def save_to_cache(data, cache_key):
    os.makedirs(os.path.dirname(cache_key), exist_ok=True)
    json_data = json.dumps(data, indent=4)
    with open(cache_key, "w") as cache_file:
        cache_file.write(json_data)


def load_from_cache(cache_key):
    if not os.path.exists(cache_key):
        return None
    try:
        with open(cache_key, "r") as cache_file:
            return json.load(cache_file)
    except json.JSONDecodeError:
        print(f"Warning: Cache file {cache_key} is corrupted or empty.")
        return None


def fetch_with_retry(repo_name, retries=5, backoff=2):
    print("I'm in fetch_with_retry")
    cache_key = generate_cache_key(repo_name)
    cached_data = load_from_cache(cache_key)

    if cached_data:
        logging.info(f"Loaded data from cache for repository: {repo_name}")
        return cached_data

    for i in range(retries):
        try:
            handle_rate_limit()  # Check rate limit before making the request
            repo = g.get_repo(repo_name)
            save_to_cache(repo.raw_data, cache_key)  # Save repository data to cache
            return repo
        except GithubException as e:
            if e.status == 403 and "rate limit" in str(e):
                handle_rate_limit()
                continue  # Retry after handling rate limit
            elif e.status == 403:
                logging.warning(
                    f"Access forbidden to repository: {repo_name}. Check your token and permissions."
                )
                return None
            elif e.status == 404:
                logging.warning(f"Repository not found: {repo_name}")
                return None
            elif isinstance(e, requests.exceptions.ReadTimeout):
                if i < retries - 1:
                    sleep_time = backoff * (2**i)
                    logging.warning(
                        f"Request timed out. Retrying in {sleep_time} seconds..."
                    )
                    time.sleep(sleep_time)
                else:
                    logging.error(f"Request timed out after {retries} retries.")
                    return None
            elif isinstance(
                e,
                (
                    NewConnectionError,
                    MaxRetryError,
                    requests.exceptions.RequestException,
                ),
            ):
                if i < retries - 1:
                    sleep_time = backoff * (2**i)
                    logging.warning(
                        f"Network error. Retrying in {sleep_time} seconds..."
                    )
                    time.sleep(sleep_time)
                else:
                    logging.error(f"Network error after {retries} retries.")
                    return None
            else:
                raise  # Re-raise exception if it's not related to rate limiting
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed: {e}")
            return None
    return None


def fetch_prs_with_retry(prs, repo, retries=5, backoff=2):
    print("I'm in fetch_prs_with_retry")
    cache_key = generate_cache_key(repo.full_name, "prs")
    cached_data = load_from_cache(cache_key)

    if cached_data:
        logging.info(f"Loaded PR data from cache for repository: {repo.full_name}")
        return cached_data

    for i in range(retries):
        try:
            handle_rate_limit()  # Check rate limit before making the request
           # prs = repo.get_pulls(state="all")
            prs_data = [pr.raw_data for pr in prs]
            save_to_cache(prs_data, cache_key)  # Save PR data to cache
            return prs
        except GithubException as e:
            if e.status == 403 and "rate limit" in str(e):
                handle_rate_limit()
                continue  # Retry after handling rate limit
            elif e.status == 403:
                logging.warning(
                    f"Access forbidden to PRs for repository: {repo.full_name}. Check your token and permissions."
                )
                return None
            elif e.status == 404:
                logging.warning(f"PRs not found for repository: {repo.full_name}.")
                return None
            elif isinstance(e, requests.exceptions.ReadTimeout):
                if i < retries - 1:
                    sleep_time = backoff * (2**i)
                    logging.warning(
                        f"Request timed out. Retrying in {sleep_time} seconds..."
                    )
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

def calculate_complexity(file_content):
    try:
        complexity = radon_complexity.cc_visit(file_content)
        return sum([item.complexity for item in complexity]) / len(complexity) if complexity else 0
    except Exception as e:
        print(f"Error calculating complexity: {e}")
        return None
def analyze_repository(repo_name):
    print("I'm in analyze_repository")
    try:
        pr_complexity_scores = []
        pr_security_issues = []
        pr_code_quality_issues = []
        logging.info(f"Analyzing repository: {repo_name}")
        handle_rate_limit()
        repo= g.get_repo(repo_name)
        fetched_repo = fetch_with_retry(repo_name)
        if not fetched_repo:
            logging.error(f"Failed to fetch repository: {repo_name}")
            return
        handle_rate_limit()
        prs = repo.get_pulls(state="all")
        fetched_prs = fetch_prs_with_retry(prs,repo)
        if prs is None:
            logging.error(f"Failed to fetch PRs for repository: {repo_name}")
            return
        #prs = repo.get_pulls(state="all")
        print("repo type: ", type(fetched_repo))
        print("repo: ", fetched_repo)
        print("pr type: ",type(fetched_prs))
        print("pr: ",fetched_prs)

        if prs is None:
            logging.error(f"Failed to fetch PRs for repository: {repo_name}")
            return

        pr_details = []
        for pr in prs:
            logging.info(f"Processing PR #{pr.number} from repository: {repo_name}")
            files = pr.get_files()
            for file in files:
                file_content = g.get_repo(repo_name).get_contents(file.filename, ref=pr.head.sha).decoded_content.decode('utf-8')
                if not file_content:
                    logging.error(f"Failed to fetch file content for : {repo_name}")
                    return
                language = detect_language(file.filename)
                if language:
                    security_issues = run_security_checks(file_content, language)
                    if security_issues:
                        pr_security_issues.append(security_issues)

                        # Analyze code quality
                    complexity_score = calculate_complexity(file_content)
                    if complexity_score is not None:
                        pr_complexity_scores.append(complexity_score)
                    code_quality_issues = analyze_code_quality(file_content, language)
                    if code_quality_issues:
                        pr_code_quality_issues.append("\n".join(code_quality_issues))
        avg_complexity_score = (
            sum(pr_complexity_scores) / len(pr_complexity_scores)
            if pr_complexity_scores
            else 0
        )
        logging.info("pr value: {pr}")
        pr_details.append(
            {
                "pr_id": pr.number,
                "repository_name": fetched_repo.name,
                "title_length": len(pr.title),
                "description_length": len(pr.body) if pr.body else 0,
                "files_changed": pr.changed_files,
                "lines_added": pr.additions,
                "lines_deleted": pr.deletions,
                "complexity_score": avg_complexity_score,
                "time_to_merge": (
                    (pr.merged_at - pr.created_at).total_seconds() / 3600
                    if pr.merged_at
                    else None
                ),
                "num_commits": pr.commits,
                "is_bug_fix": "bug" in [label.name for label in pr.get_labels()],
                "num_comments": pr.comments + pr.review_comments,
                "num_reviewers": len(
                    set([review.user.login for review in pr.get_reviews()])
                ),
                "merge_status": pr.merged,
                "author_experience": fetched_repo.get_commits(author=pr.user.login).totalCount,
                "author_tenure": (pr.created_at - pr.user.created_at).days / 30,
                "author_contributions": pr.user.contributions,
                "review_time": sum(
                    [
                        (review.submitted_at - pr.created_at).total_seconds() / 3600
                        for review in pr.get_reviews()
                        if review.submitted_at
                    ]
                ),
                "num_approvals": len(
                    [
                        review
                        for review in pr.get_reviews()
                        if review.state == "APPROVED"
                    ]
                ),
                "review_sentiment_score": sum(
                    [
                        calculate_sentiment(comment.body)
                        for comment in pr.get_issue_comments()
                    ]
                ),
                "security_issues": "\n".join(
                    pr_security_issues
                ),  # Concatenate all security issues found
                "code_quality_issues": "\n".join(
                    pr_code_quality_issues
                ),  # Concatenate all code quality issues found
            }
        )
        logging.info(f"Pull Request Details: {pr_details}")
        return pr_details
    except Exception as e:
        logging.error(f"errors details: {e}")
        logging.error(f"Error processing PR in {repo_name}")
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

def main():
    all_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_repo = {
            executor.submit(analyze_repository, repo): repo for repo in repositories
        }
        for future in concurrent.futures.as_completed(future_to_repo):
            repo = future_to_repo[future]
            try:
                repo_results = future.result()
                if repo_results:
                    all_results.append(repo_results)
            except Exception as exc:
                logging.error(f"Repository {repo} generated an exception: {exc}")

    # Convert the results to a DataFrame
    print("before save pr_data: ", all_results)
    if all_results:
        df = pd.DataFrame(all_results)
        print("DataFrame created successfully.")
        print(df.head())  # Check the first few rows of the DataFrame

        # Export the DataFrame to a CSV file
        
        output_file = 'repo_analysis_results.csv'
        try:
            if check_if_csv_exists(output_file):
                append_to_csv(all_results,output_file)
                print(f"Dataset updated to {output_file}")
            else:
                df.to_csv(output_file, index=False)
                print(f"Dataset saved to {output_file}")
        except Exception as e:
            print(f"Failed to save dataset: {e}")
    else:
        print("No PR data to save.")

        df = pd.DataFrame(all_results)
        df.to_csv("repo_analysis_results.csv", index=False)
        logging.info("Analysis complete. Results saved to repo_analysis_results.csv")


if __name__ == "__main__":
    main()
