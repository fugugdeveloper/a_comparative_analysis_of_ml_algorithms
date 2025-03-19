from github import Github
import os

ACCESS_TOKEN = "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"
g = Github(ACCESS_TOKEN)

# Access the repository
repo_name = "facebook/react"  # Replace with actual repo name
repo = g.get_repo(repo_name)
pr_data = []

pull_requests = repo.get_pulls(state='all')  # Get all pull requests (open, closed, merged)
for pr in pull_requests:
    
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
        "num_reviewers": len(set([review.user.login for review in pr.get_reviews()])),
        "merge_status": pr.merged,
        "author_experience": repo.get_commits(author=pr.user.login).totalCount,
        "author_tenure": (pr.created_at - pr.user.created_at).days / 30,
        "author_contributions": pr.user.contributions,
        "review_time": sum([(review.submitted_at - pr.created_at).total_seconds() / 3600 for review in pr.get_reviews()]),
        "num_approvals": len([review for review in pr.get_reviews() if review.state == 'APPROVED']),
    }

    pr_data.append(pr_details)

import pandas as pd
df = pd.DataFrame(pr_data)
df.to_csv("pull_request_dataset.csv")