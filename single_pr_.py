import os
from github import Github
import pandas as pd
from textblob import TextBlob
import radon.complexity as radon_complexity
import requests

# Set up your GitHub personal access token
ACCESS_TOKEN = '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'

# Authenticate to GitHub
g = Github(ACCESS_TOKEN)

# List of repositories you want to analyze
repositories = [
    "facebook/react",
    "microsoft/vscode",
    "django/django",
    "spring-projects/spring-framework",
    'opencv/opencv'
]
 # Add more repository names as needed

# Function to calculate sentiment of review comments
def calculate_sentiment(comment):
    return TextBlob(comment).sentiment.polarity

# Function to calculate the complexity score of a file using radon
def calculate_complexity(file_content):
    try:
        complexity = radon_complexity.cc_visit(file_content)
        return sum([item.complexity for item in complexity]) / len(complexity) if complexity else 0
    except Exception as e:
        print(f"Error calculating complexity: {e}")
        return None

# Function to fetch a file's content from GitHub
def get_file_content(repo, file_path):
    try:
        file_content = repo.get_contents(file_path).decoded_content.decode('utf-8')
        return file_content
    except Exception as e:
        print(f"Error fetching file content for {file_path}: {e}")
        return None

# Initialize a list to hold PR data
pr_data = []

# Loop through repositories
for repo_name in repositories:
    try:
        repo = g.get_repo(repo_name)
    except Exception as e:
        print(f"Error accessing repository {repo_name}: {e}")
        continue  # Skip to the next repository if there's an error

    # Loop through pull requests in the repository
    try:
        pull_requests = repo.get_pulls(state='all')  # Get all PRs (open, closed, merged)
        for pr in pull_requests:
            try:
                # Calculate complexity score for the PR
                complexity_scores = []
                for file in pr.get_files():
                    if file.filename.endswith('.py'):  # Adjust this condition for non-Python projects
                        file_content = get_file_content(repo, file.filename)
                        if file_content:
                            complexity_score = calculate_complexity(file_content)
                            if complexity_score is not None:
                                complexity_scores.append(complexity_score)

                # Average complexity score for the PR
                avg_complexity_score = sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0

                pr_details = {
                    "pr_id": pr.number,
                    "repository_name": repo_name,
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
                    "author_experience": len(repo.get_commits(author=pr.user.login)),
                    "author_tenure": (pr.created_at - pr.user.created_at).days / 30,
                    "author_contributions": pr.user.contributions,
                    "review_time": sum([(review.submitted_at - pr.created_at).total_seconds() / 3600 for review in pr.get_reviews()]),
                    "num_approvals": len([review for review in pr.get_reviews() if review.state == 'APPROVED']),
                    "review_sentiment_score": sum([calculate_sentiment(comment.body) for comment in pr.get_issue_comments()]),
                }

                pr_data.append(pr_details)

            except Exception as e:
                print(f"Error processing PR {pr.number} in {repo_name}: {e}")
                continue  # Skip to the next PR if there's an error

        # Calculate repository-wide features
        repo_activity_level = len(list(repo.get_commits()))
        repo_size = repo.size / 1024  # Convert repository size from KB to MB

        for pr in pr_data:
            pr['repo_activity_level'] = repo_activity_level
            pr['repo_size'] = repo_size

    except Exception as e:
        print(f"Error processing repository {repo_name}: {e}")
        continue  # Skip to the next repository if there's an error

# Convert the list of dictionaries to a pandas DataFrame
df = pd.DataFrame(pr_data)

# Export the DataFrame to a CSV file
output_file = 'pr_dataset.csv'
df.to_csv(output_file, index=False)

print(f"Dataset saved to {output_file}")
