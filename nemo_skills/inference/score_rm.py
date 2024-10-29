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

import json
import logging
import sys

# from nemo_skills.code_execution.sandbox import sandbox_params
# from nemo_skills.inference.server.code_execution_model import (
#     ErrorRecoveryConfig,
#     server_params,
# )
# from nemo_skills.utils import get_fields_docstring, get_help_message, nested_dataclass, setup_logging
from dataclasses import asdict, dataclass, field

import hydra
import numpy as np
from pytriton.client import ModelClient
from torch.utils.data import DataLoader, Dataset

LOG = logging.getLogger(__file__)


@dataclass
class RewardModelGenerationConfig:
    input_file: str
    output_file: str
    batch_size: int = 32
    server_address: str = "localhost:5000"

    def __post_init__(self):
        # TODO: input validations
        pass


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_reward_model_generation_config", node=RewardModelGenerationConfig)


class JSONLDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path) as reader:
            for l in reader:
                obj = json.loads(l)
                for k, v in obj.items():
                    if v is None:
                        obj[k] = ''
                self.data.append(obj)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {'idx': idx, **self.data[idx]}


def query_reward_model(input_file, output_file, batch_size=32, server_url="localhost:5555"):
    client = ModelClient(server_url, "reward_model")
    # Create dataset and dataloader
    dataset = JSONLDataset(input_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Process data in batches
    with open(output_file, mode='w') as writer:
        for batch in dataloader:
            sentences = np.array([[obj.encode('utf-8')] for obj in batch['generation']], dtype=np.bytes_)
            result = client.infer_batch(sentences=sentences)
            for i, idx in enumerate(batch['idx']):
                out_obj = dataset.data[idx]
                out_obj.update({'reward_model_score': result['rewards'][i].item()})
                writer.write(json.dumps(out_obj) + '\n')


@hydra.main(version_base=None, config_name='base_reward_model_generation_config')
def score_rm(cfg: RewardModelGenerationConfig):
    cfg = RewardModelGenerationConfig(**cfg)  # _init_nested=True, **cfg)

    LOG.info("Config used: %s", cfg)

    # TODO this should probably look more like generate at some point with per-server clients
    # encapsulated, however, that complexity is not needed quite yet.
    query_reward_model(cfg.input_file, cfg.output_file, cfg.batch_size, cfg.server_address)


# error_recovery_params = '\n' + get_fields_docstring(
#     ErrorRecoveryConfig,
#     prefix='server.error_recovery.',
#     level=2,
# )

# TODO: just avoiding this right now so I can run this script in nemo container isntead of nemo skills container
HELP_MESSAGE = "help"  # get_help_message(
# RewardModelGenerationConfig,
# server_params=server_params(),
# sandbox_params=sandbox_params(),
# error_recovery_params=error_recovery_params,
# )

if __name__ == "__main__":
    if '--help' in sys.argv or '-h' in sys.argv:
        print(HELP_MESSAGE)
    else:
        # setup_logging()
        score_rm()
