#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
from contextlib import AsyncExitStack
from typing import Any, Callable

from mcp import ClientSession
from mcp.client.sse import sse_client

from .config import MCP_SERVER_URL

async def connect_mcp(stack: AsyncExitStack) -> ClientSession:
    transport = await stack.enter_async_context(sse_client(MCP_SERVER_URL))
    session = await stack.enter_async_context(ClientSession(transport[0], transport[1]))
    await session.initialize()
    return session

async def call_tool_with_retry(
    get_session: Callable[..., Any],
    tool_name: str,
    tool_query: str,
    timeout: int = 60,
    retries: int = 3
):
    for i in range(retries):
        try:
            session = await get_session()
            return await asyncio.wait_for(
                session.call_tool(tool_name, {"query": tool_query}),
                timeout=timeout
            )
        except (asyncio.TimeoutError, Exception):
            if i == retries - 1:
                raise
            await asyncio.sleep(0.5 * (2 ** i))
            await get_session(reconnect=True)
