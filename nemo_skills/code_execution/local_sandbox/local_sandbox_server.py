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


# Function to execute Python code in a subprocess
def execute_python(generated_code, timeout):
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=execute_code_subprocess, args=(generated_code, queue))
    process.start()
    process.join(timeout=timeout)

    if process.is_alive():  # If the process didn't finish within the timeout
        process.kill()
        return {"process_status": "timeout", "stdout": "Timed out", "stderr": "Timed out"}
    
    return queue.get()


# Function to execute Lean4 code using subprocess and a temporary file
def execute_lean4(generated_code, timeout):
    # Create a temporary file to store Lean4 code
    with tempfile.NamedTemporaryFile(delete=False, suffix=".lean") as temp_file:
        temp_file_name = temp_file.name
        temp_file.write(generated_code.encode('utf-8'))

    try:
        # Run Lean4 code using subprocess and the temporary file
        result = subprocess.run(
            ["lean", temp_file_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout
        )

        # Return the result from stdout and stderr
        return {
            "process_status": "finished", # could be replaced by 0
            "stdout": result.stdout.decode('utf-8'),
            "stderr": result.stderr.decode('utf-8')
        }
    except subprocess.TimeoutExpired:
        # Handle the timeout case
        return {"process_status": "timeout", "stdout": "Timed out", "stderr": "Timed out"}
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_name):
            os.remove(temp_file_name)


# Function that runs Python code in a subprocess and limits resources
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
    generated_code = request.json['generated_code']
    timeout = request.json['timeout']
    language = request.json.get('language', 'python')  

    if language == 'python':
        return execute_python(generated_code, timeout)
    elif language == 'lean4':
        return execute_lean4(generated_code, timeout)


if __name__ == "__main__":
    app.run(debug=True)
