#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

def setup_logger(name: str = "eval_agent_pipeline") -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    return logging.getLogger(name)
