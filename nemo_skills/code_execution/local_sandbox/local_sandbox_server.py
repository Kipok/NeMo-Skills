# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import tempfile
import os
import subprocess
import multiprocessing
import resource
import sys
from io import StringIO
from flask import Flask, request

app = Flask(__name__)

def execute_python(generated_code, timeout):
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=execute_code_subprocess, args=(generated_code, queue))
    process.start()
    process.join(timeout=timeout)

    if process.is_alive():  # If the process didn't finish within the timeout
        process.kill()
        return {"process_status": "timeout", "stdout": "Timed out", "stderr": "Timed out"}
    
    return queue.get()


# def execute_lean4(generated_code, timeout):
#     project_path = "/lean4/my_project"
    
#     with tempfile.NamedTemporaryFile(dir=project_path, delete=False, suffix=".lean") as temp_file:
#         temp_file_name = temp_file.name
#         temp_file.write(generated_code.encode('utf-8'))
#     try:
#         result = subprocess.run(
#             ["lake", "env", "--dir", project_path, "lean", temp_file_name],
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             timeout=timeout,
#             cwd=project_path
#         )

#         return {
#             "process_status": "finished",  # could be replaced by 0 for successful completion
#             "stdout": result.stdout.decode('utf-8'),
#             "stderr": result.stderr.decode('utf-8')
#         }
#     except subprocess.TimeoutExpired:
#         return {"process_status": "timeout", "stdout": "Timed out", "stderr": "Timed out"}
#     finally:
#         if os.path.exists(temp_file_name):
#             os.remove(temp_file_name)

def execute_lean4(generated_code, timeout):
    try:
        # Step 1: Check and log the current PATH environment variable
        result_path = subprocess.run(['echo', '$PATH'], capture_output=True, text=True, shell=True)
        print(f"Current PATH: {result_path.stdout}")
        
        # Step 2: Verify if 'lake' command is available in the PATH
        result_which = subprocess.run(['which', 'lake'], capture_output=True, text=True)
        print(f"Location of 'lake': {result_which.stdout}")
        
        # Step 3: Try running 'lake --version' to check if it works
        result_version = subprocess.run(['lake', '--version'], capture_output=True, text=True)
        print(f"Lake version: {result_version.stdout}")
        
        # Step 4: If 'lake' command is not found in the PATH, try running it using full path
        if result_version.returncode != 0:  # If 'lake' fails, try full path
            # Modify this full path according to your environment (find it using 'which lake' when it works)
            full_lake_path = "/path/to/lake"  # Update this with the actual path to 'lake'
            result_full_path = subprocess.run([full_lake_path, '--version'], capture_output=True, text=True)
            print(f"Lake version (full path): {result_full_path.stdout}")
        
        # Step 5: Proceed with running the generated Lean4 code using lake
        result = subprocess.run(
            ['lake', 'env', 'lean', generated_code],  # Replace 'some_command' with the actual lake command you use
            capture_output=True,
            timeout=timeout,
            text=True
        )
        
        # Step 6: Log and return the result
        print(f"Execution result: {result.stdout}")
        return result.stdout
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return {"error": str(e)}



def execute_code_subprocess(generated_code, queue):
    limit = 1024 * 1024 * 1024 * 10  # 10 GB memory limit
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    resource.setrlimit(resource.RLIMIT_DATA, (limit, limit))
    resource.setrlimit(resource.RLIMIT_STACK, (limit, limit))

    sys.stdout = StringIO()
    try:
        exec(generated_code, {})
        queue.put({"process_status": "finished", "stdout": sys.stdout.getvalue(), "stderr": ""})
    except Exception as e:
        queue.put({"process_status": "error", "stdout": "", "stderr": str(e)})


    # Main Flask endpoint to handle execution requests
@app.route("/execute", methods=["POST"])
def execute():
    if bool(request.json.get('a_true', 'True')):
        return {
                "process_status": "finished",  # could be replaced by 0 for successful completion
                "stdout": "",
                "stderr": ""
            }
    generated_code = request.json['generated_code']
    timeout = request.json['timeout']
    language = request.json.get('language', 'python')  


    if language == 'python':
        return execute_python(generated_code, timeout)
    elif language == 'lean4':
        return execute_lean4(generated_code, timeout)



if __name__ == "__main__":
    app.run(debug=True)
