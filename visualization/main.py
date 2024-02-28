import logging
from pathlib import Path
import sys

import hydra

sys.path.append(str(Path(__file__).parent.parent))

from callbacks import app
from layouts import get_main_page_layout
from settings.config import ConfigHolder, Config
import signal

signal.signal(signal.SIGALRM, signal.SIG_IGN)

logging.getLogger().setLevel(logging.INFO)


@hydra.main(version_base=None, config_path="settings", config_name="config")
def set_config(cfg: Config) -> None:
    ConfigHolder.initialize(cfg)


if __name__ == "__main__":
    set_config()
    logging.info(f"Config initialized {ConfigHolder.get_config()}")
    app.layout = get_main_page_layout()
    app.run_server(
        host='localhost',
        port='8080',
    )
