from dataclasses import dataclass, field
from typing import Dict, Optional

import hydra
from omegaconf import DictConfig, OmegaConf

from nemo_skills.inference.generate_solutions import (
    GenerateSolutionsConfig,
)

from nemo_skills.utils import unroll_files


@dataclass
class BaseVisualizationConfig:
    prediction_jsonl_files: Dict[str, str] = field(default_factory=dict)

    code_separators: str = "<llm-code>\n \n</llm-code>"
    code_output_separators: str = "<llm-code-output>\n \n</llm-code-output>"
    dataset_path: Optional[str] = "datasets/{}/{}.jsonl"
    save_dataset_path: Optional[str] = "results/saved_dataset"


@dataclass
class Config(GenerateSolutionsConfig):
    visualization_params: BaseVisualizationConfig = field(
        default_factory=BaseVisualizationConfig
    )


class ConfigHolder:
    _config = None

    @staticmethod
    def initialize(cfg: DictConfig) -> None:
        config_dict = OmegaConf.to_container(cfg)
        config_dict['visualization_params']['prediction_jsonl_files'] = {
            model_name: list(unroll_files(file_path.split(" ")))
            for model_name, file_path in config_dict['visualization_params'][
                'prediction_jsonl_files'
            ].items()
        }
        config_dict['inference']['start_random_seed'] = (
            config_dict['inference']['start_random_seed']
            if 'start_random_seed' in config_dict['inference']
            else 0
        )
        ConfigHolder._config = config_dict

    @staticmethod
    def get_config() -> Dict:
        if ConfigHolder._config is None:
            raise RuntimeError("Config has not been initialized")
        return ConfigHolder._config


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="config", node=Config)
