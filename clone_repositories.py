import subprocess
import os

# Directory to clone repositories into
CLONE_DIR = 'cloned_repos'

def clone_repo(repo_url, clone_dir):
    if not os.path.exists(clone_dir):
        os.makedirs(clone_dir)
    repo_name = repo_url.split('/')[-1]
    repo_path = os.path.join(clone_dir, repo_name)
    if not os.path.exists(repo_path):
        subprocess.run(['git', 'clone', repo_url, repo_path])
    else:
        print(f"Repository {repo_name} already cloned.")
    return repo_path
