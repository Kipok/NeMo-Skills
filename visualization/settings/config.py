from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import hydra

from nemo_skills.code_execution.utils import CODE_OUTPUT_SEPARATORS, CODE_SEPARATORS
from nemo_skills.inference.generate_solutions import GenerateSolutionsConfig


@dataclass
class BaseVisualizationConfig:
    prediction_jsonl_files: Dict[str, str] = field(default_factory=dict)

    code_separators: Tuple[str, str] = CODE_SEPARATORS
    code_output_separators: Tuple[str, str] = CODE_OUTPUT_SEPARATORS
    dataset_path: Optional[str] = "datasets/{}/{}.jsonl"
    save_dataset_path: Optional[str] = "results/saved_dataset"


@dataclass
class Config(GenerateSolutionsConfig):
    visualization_params: BaseVisualizationConfig = field(default_factory=BaseVisualizationConfig)


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="config", node=Config)
