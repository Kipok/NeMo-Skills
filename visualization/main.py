import logging
import sys
from pathlib import Path

import hydra

sys.path.append(str(Path(__file__).parent.parent))

import signal

from callbacks import app
from layouts import get_main_page_layout

signal.signal(signal.SIGALRM, signal.SIG_IGN)

logging.getLogger().setLevel(logging.INFO)


if __name__ == "__main__":
    app.layout = get_main_page_layout()
    app.run(
        host='localhost',
        port='8080',
    )
