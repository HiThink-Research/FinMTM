#!/usr/bin/env python3
"""Runtime configuration for financial-agent evaluation."""

import os


MCP_SERVER_URL = os.getenv("FINMTM_MCP_URL", "http://localhost:8081/sse")
MAX_ITER = int(os.getenv("FINMTM_MAX_ITER", "8"))
