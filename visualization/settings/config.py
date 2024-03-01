import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import hydra

from nemo_skills.inference.generate_solutions import GenerateSolutionsConfig
from nemo_skills.utils import unroll_files


@dataclass
class BaseVisualizationConfig:
    prediction_jsonl_files: Dict[str, str] = field(default_factory=dict)

    code_separators: str = "<llm-code>\n \n</llm-code>"
    code_output_separators: str = "<llm-code-output>\n \n</llm-code-output>"
    dataset_path: Optional[str] = "datasets/{}/{}.jsonl"
    save_dataset_path: Optional[str] = "results/saved_dataset"

    # def __post_init__(self):
    #     """Building data_file from dataset/split_name if not provided directly."""
    #     if isinstance(self.prediction_jsonl_files, str):
    #         self.prediction_jsonl_files = {
    #             model_name: list(unroll_files(file_path.split(" ")))
    #             for model_name, file_path in self.prediction_jsonl_files.items()
    #         }


@dataclass
class Config(GenerateSolutionsConfig):
    visualization_params: BaseVisualizationConfig = field(default_factory=BaseVisualizationConfig)


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="config", node=Config)
