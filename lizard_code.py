import subprocess
import tempfile
import os
import sys

def extract_lizard_metrics(file_content, language_ext='.java'):
    try:
        python_executable = sys.executable
        if isinstance(file_content, tuple):
            file_content = ''.join(file_content)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix=language_ext, delete=False) as temp_file:
            temp_filename = temp_file.name
            temp_file.write(file_content)
            temp_file.flush()
        
        # Run Lizard on the temporary file
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
        
    except Exception as e:
        print(f"Error running Lizard: {e}")
        return (None, None, None, None, None, None, None, None)
    finally:
        # Ensure the temporary file is deleted after processing
        try:
            os.remove(temp_filename)
        except Exception as e:
            print(f"Error removing temp file {temp_filename}: {e}")

# Example usage
file_content = """public class Example {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
    
    public int add(int a, int b) {
        return a + b;
    }
}
"""
NLOC, Avg_NLOC, AvgCCN, Avg_token, Fun_Cnt, Warning_cnt, Fun_Rt, nloc_Rt = extract_lizard_metrics(file_content)
print("NLOC: ", NLOC)
print("Avg.NLOC: ", Avg_NLOC)
print("AvgCCN: ", AvgCCN)
print("Avg.token: ", Avg_token)
print("Fun Cnt: ", Fun_Cnt)
print("Warning cnt: ", Warning_cnt)
print("Fun Rt: ", Fun_Rt)
print("nloc Rt: ", nloc_Rt)
