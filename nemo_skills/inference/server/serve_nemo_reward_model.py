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

import logging
from dataclasses import dataclass

import hydra
import numpy as np
from flask import Flask, jsonify, request
from pytriton.client import ModelClient

LOG = logging.getLogger(__file__)


@dataclass
class RewardModelGenerationConfig:
    inference_port: int = 5000
    triton_server_address: str = "localhost:5001"


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_reward_model_generation_config", node=RewardModelGenerationConfig)


@hydra.main(version_base=None, config_name="base_reward_model_generation_config")
def proxy_rm(cfg: RewardModelGenerationConfig) -> None:
    app = Flask(__name__)

    client = ModelClient(cfg.triton_server_address, "reward_model")

    @app.route('/score', methods=['POST'])
    def infer():
        data = request.json
        input_data = np.array([[obj.encode('utf-8')] for obj in data['prompts']], dtype=np.bytes_)

        result = client.infer_batch(sentences=input_data)

        json_output = jsonify({'rewards': result['rewards'][0].tolist()})
        return json_output

    app.run(host='127.0.0.1', port=cfg.inference_port, threaded=False)


if __name__ == "__main__":
    proxy_rm()
