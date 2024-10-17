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


import multiprocessing
import os
import resource
import subprocess
import sys
import tempfile
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

def execute_lean4(generated_code, timeout):
    temp_file_name = None
    try:
        elan_path = "/root/.elan/bin"
        os.environ['PATH'] = elan_path + ":" + os.environ['PATH']

        project_path = "/lean4/my_project"
        with tempfile.NamedTemporaryFile(dir=project_path, delete=False, suffix=".lean") as temp_file:
            temp_file_name = temp_file.name
            temp_file.write(generated_code.encode('utf-8'))

        result = subprocess.run(
            ['lake', 'env', '--dir', project_path, 'lean', temp_file_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            cwd=project_path,  # Ensure we are in the correct working directory
        )

        if result.returncode == 0:
            process_status = "completed"
        else:
            process_status = "failed"

        return {
            "process_status": process_status,
            "stdout": result.stdout.decode('utf-8'),
            "stderr": result.stderr.decode('utf-8'),
        }

    except subprocess.TimeoutExpired:
        return {"process_status": "timeout", "stdout": "Timed out", "stderr": "Timed out"}
    except Exception as e:
        print(f"Error: {str(e)}")
        return {"process_status": "error", "stdout": "", "stderr": str(e)}
    finally:
        # Safely remove the temporary file if it was created
        if temp_file_name and os.path.exists(temp_file_name):
            os.remove(temp_file_name)


def execute_code_subprocess(generated_code, queue):
    limit = 1024 * 1024 * 1024 * 10  # 10 GB memory limit
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    resource.setrlimit(resource.RLIMIT_DATA, (limit, limit))
    resource.setrlimit(resource.RLIMIT_STACK, (limit, limit))

    sys.stdout = StringIO()
    try:
        exec(generated_code, {})
        queue.put(sys.stdout.getvalue())
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


