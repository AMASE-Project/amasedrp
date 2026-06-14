#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File:         logging.py
@Author:       Guangquan ZENG
@Contact:      guangquan.zeng@outlook.com
@Description:  Logging utilities for the data reduction pipeline (DRP).
"""

import logging
import os


def configure_logging(
    log_file: str,
    level: int = logging.INFO,
    fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
) -> None:
    """Configure file-based logging for the DRP.

    Attaches a ``FileHandler`` to the root ``amasedrp`` logger so that
    *INFO* and *WARNING* messages (and anything at or above *level*)
    from any ``amasedrp`` submodule are persisted to disk. Calling the
    function multiple times with the same *log_file* is a no-op.

    Args:
        log_file: Path to the log file. The directory is created if it
            does not exist.
        level: Minimum logging level to emit (default ``logging.INFO``).
        fmt: Log record format string.
    """
    # Configure the root logger for the data reduction pipeline
    root_logger = logging.getLogger("amasedrp")
    root_logger.setLevel(level)

    abs_log_file = os.path.abspath(os.path.expanduser(log_file))
    # Calling this function multiple times with the same log file is a no-op
    for handler in root_logger.handlers:
        if (
            isinstance(handler, logging.FileHandler)
            and handler.baseFilename == abs_log_file
        ):
            return

    log_dir: str = os.path.dirname(abs_log_file)
    if log_dir and not os.path.isdir(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    handler = logging.FileHandler(abs_log_file, mode="a")
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt)
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
