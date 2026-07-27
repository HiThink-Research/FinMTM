#!/usr/bin/env python3
"""MCP client adapter used by the ReAct-style financial agent."""

import asyncio
from contextlib import AsyncExitStack
from typing import Any, Callable

from mcp import ClientSession
from mcp.client.sse import sse_client

from . import config


async def connect_mcp(stack: AsyncExitStack) -> ClientSession:
    transport = await stack.enter_async_context(
        sse_client(config.MCP_SERVER_URL)
    )
    session = await stack.enter_async_context(
        ClientSession(transport[0], transport[1])
    )
    await session.initialize()
    return session


async def call_tool_with_retry(
    get_session: Callable[..., Any],
    tool_name: str,
    tool_query: str,
    timeout: int = 60,
    retries: int = 3,
):
    for attempt in range(retries):
        try:
            session = await get_session()
            return await asyncio.wait_for(
                session.call_tool(tool_name, {"query": tool_query}),
                timeout=timeout,
            )
        except Exception:
            if attempt == retries - 1:
                raise
            await asyncio.sleep(0.5 * (2**attempt))
            await get_session(reconnect=True)
