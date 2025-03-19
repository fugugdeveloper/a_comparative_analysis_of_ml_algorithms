import json
import subprocess

def get_pylint_metrics(file_path):
    try:
        result = subprocess.run(['pylint', file_path, '-f', 'json'], capture_output=True, text=True, check=True)
        pylint_output = json.loads(result.stdout)
        metrics = {
            'num_errors': sum(1 for issue in pylint_output if issue['type'] == 'error'),
            'num_warnings': sum(1 for issue in pylint_output if issue['type'] == 'warning'),
            'num_refactors': sum(1 for issue in pylint_output if issue['type'] == 'refactor'),
            'num_convention_issues': sum(1 for issue in pylint_output if issue['type'] == 'convention')
        }
    except subprocess.CalledProcessError:
        metrics = {
            'num_errors': 0,
            'num_warnings': 0,
            'num_refactors': 0,
            'num_convention_issues': 0
        }
    return metrics
