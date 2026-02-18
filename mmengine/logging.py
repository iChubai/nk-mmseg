"""Logging compatibility layer."""

from __future__ import annotations

import logging
from typing import Optional


class MMLogger(logging.Logger):
    _current = None

    @classmethod
    def get_current_instance(cls) -> 'MMLogger':
        if cls._current is None:
            logger = logging.getLogger('mmengine')
            if not logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter(
                    '[%(asctime)s] %(name)s %(levelname)s: %(message)s'))
                logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            cls._current = logger
        return cls._current


def print_log(msg, logger=None, level=logging.INFO):
    if logger is None:
        MMLogger.get_current_instance().log(level, msg)
    elif isinstance(logger, str):
        logging.getLogger(logger).log(level, msg)
    else:
        logger.log(level, msg)
