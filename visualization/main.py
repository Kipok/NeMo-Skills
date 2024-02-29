import logging
from pathlib import Path
import sys

import hydra

sys.path.append(str(Path(__file__).parent.parent))

from callbacks import app
from layouts import get_main_page_layout

import signal

signal.signal(signal.SIGALRM, signal.SIG_IGN)

logging.getLogger().setLevel(logging.INFO)


if __name__ == "__main__":
    app.layout = get_main_page_layout()
    app.run(
        host='localhost',
        port='8080',
    )
