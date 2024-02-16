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
import resource
import traceback

from flask import Flask, jsonify, request
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils import io

from nemo_skills.code_execution.math_grader import math_equal
from nemo_skills.code_execution.sandbox import Sandbox

app = Flask(__name__)


# need to memory-limit to avoid common errors of allocating too much
# but this has to be done in a subprocess to not crush server itself
def execute_code_subprocess(generated_code, state, max_output_characters, queue):
    # we are going to re-run all previous code snippets
    # in separate ipython cells to restore the state
    code_snippets = state + [generated_code]

    # this can be overriden inside run_cell, so it's not a guaranteed protection
    limit = 1024 * 1024 * 1024 * 10  # 10gb - somehow with a smaller limit the server dies when numpy is used
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    resource.setrlimit(resource.RLIMIT_DATA, (limit, limit))
    resource.setrlimit(resource.RLIMIT_STACK, (limit, limit))

    try:
        shell = InteractiveShell()
        for code in code_snippets:
            with io.capture_output() as captured:
                exec_result = shell.run_cell(code)
        # serializing to str to make sure things like Rational can be converted to json
        output = f"{captured.stdout}{captured.stderr}".strip().replace("Out[1]: ", "")
        if len(output) > max_output_characters:
            output = output[:max_output_characters] + "<output cut>"
        error_message = ""
        state = code_snippets
        if exec_result.error_in_exec is not None:
            # full traceback will be part of output
            error_message = f"{Sandbox.EXECUTION_ERROR} {str(exec_result.error_in_exec)}"
            # resetting state to not re-execute code with errors
            state = []
        elif exec_result.error_before_exec is not None:
            # full traceback will be part of output
            error_message = f"{Sandbox.SYNTAX_ERROR} {str(exec_result.error_before_exec)}"
            # resetting state to not re-execute code with errors
            state = []
        elif output == "":
            error_message = Sandbox.RESULT_NOT_DEFINED_ERROR
        queue.put({"result": output, "error_message": error_message, "state": state})
    except Exception:
        # removing useless prefix from traceback
        queue.put(
            {
                "result": None,
                "error_message": Sandbox.UNDEFINED_ERROR + "\n".join(traceback.format_exc().split("\n")[3:]),
                "state": [],
            }
        )


@app.route("/execute_code", methods=["PUT"])
def execute_code():
    state = request.json['state']
    generated_code = request.json['generated_code']
    timeout = request.json['timeout']
    max_output_characters = request.json['max_output_characters']
    # running in a separate process to ensure any kind of crashes are properly handled
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=execute_code_subprocess, args=(generated_code, state, max_output_characters, queue)
    )
    process.start()
    process.join(timeout=timeout)
    if process.is_alive():  # didn't finish successfully
        process.kill()
        return {"result": None, "error_message": Sandbox.TIMEOUT_ERROR, "state": []}
    return queue.get()


@app.route("/is_output_correct", methods=["PUT"])
def is_output_correct():
    pred_output = request.json["pred_output"]
    gt_output = request.json["gt_output"]
    include_percentage = request.json["include_percentage"]
    tolerance = request.json["tolerance"]
    timeout = request.json["timeout"]
    return jsonify(math_equal(pred_output, gt_output, include_percentage, tolerance, timeout))
