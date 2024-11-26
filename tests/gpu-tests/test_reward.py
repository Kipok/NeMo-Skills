import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).absolute().parents[1]))
from test_data_preparation import docker_rm_and_mkdir


@pytest.mark.gpu
def test_vllm_reward():
    # input_file = "/tmp/nemo-skills-data/tests/data/output-rs0.test"
    # output_file = "/tmp/nemo-skills-data/tests/data/rm-output-rs0.jsonl"

    # test_config_path = Path(__file__).absolute().parent / "test-local.yaml"
    # config = yaml.safe_load(open(test_config_path).read())
    # volumes = config['mounts']
    # print(volumes)
    # container = config['containers']['nemo-skills']
    # docker_run(
    #     image_name=container,
    #     volume_paths=volumes,
    #     command=f'mkdir -p {Path(input_file).parent} && cp tests/data/output-rs0.test {input_file}',
    # )

    model_path = os.getenv('NEMO_SKILLS_TEST_HF_MODEL')
    if not model_path:
        pytest.skip("Define NEMO_SKILLS_TEST_HF_MODEL to run this test")
    model_type = os.getenv('NEMO_SKILLS_TEST_MODEL_TYPE')
    if not model_type:
        pytest.skip("Define NEMO_SKILLS_TEST_MODEL_TYPE to run this test")
    prompt_template = 'llama3-instruct' if model_type == 'llama' else 'qwen-instruct'

    input_file = "/nemo_run/code/tests/data/output-rs0.test"
    output_file = "/tmp/nemo-skills-tests/data/rm-output-rs0.jsonl"

    docker_rm_and_mkdir(output_file)

    cmd = (
        f"ns generate "
        f"    --cluster test-local --config_dir {Path(__file__).absolute().parent} "
        f"    --model {model_path} "
        f"    --server_type vllm "
        f"    --generation_type reward "
        f"    --server_gpus 1 "
        f"    --server_nodes 1 "
        f"    ++input_file={input_file} "
        f"    ++output_file={output_file} "
        f"    --output_dir={os.path.dirname(output_file)} "
        f"    ++prompt_config=generic/math "
        f"    ++prompt_template={prompt_template} "
        f"    ++batch_size=8 "
        f"    ++max_samples=10 "
        f"    ++skip_filled=False "
    )
    subprocess.run(cmd, shell=True, check=True)

    # no evaluation by default - checking just the number of lines and that there is a "reward_model_score" key
    with open(output_file) as fin:
        lines = fin.readlines()
    assert len(lines) == 10
    for line in lines:
        data = json.loads(line)
        assert 'reward_model_score' in data
