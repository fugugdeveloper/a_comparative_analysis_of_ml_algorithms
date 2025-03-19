from github import Github

ACCESS_TOKEN = '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'

# Authenticate to GitHub
g = Github(ACCESS_TOKEN)
rate_limit = g.get_rate_limit()
print("Rate_Limit: ",rate_limit)