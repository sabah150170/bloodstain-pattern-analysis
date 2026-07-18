
# SPDX-FileCopyrightText: 2026 Buse Nur Sabah
# SPDX-License-Identifier: GPL-3.0-only


import os
import logging
import sys
from config import DEBUG_MODE


def setup_logger(run_dir):
    logger = logging.getLogger(run_dir)
    logger.setLevel(logging.DEBUG)
    logger.setLevel(logging.CRITICAL)

    logger.propagate = False

    if logger.handlers:
        return logger

    if DEBUG_MODE:
        os.makedirs(run_dir, exist_ok=True)
        log_path = os.path.join(run_dir, "run.log")
        handler = logging.FileHandler(log_path)
        fmt = logging.Formatter("%(levelname)s%(pipe)s%(message)s")
        handler.setFormatter(fmt)

        def _filter(record):
            record.pipe = " | " if record.getMessage() else ""
            if not record.getMessage():
                record.levelname = ""
            return True

        handler.addFilter(type("F", (logging.Filter,), {"filter": staticmethod(_filter)})())
    else:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))

    logger.addHandler(handler)
    return logger
