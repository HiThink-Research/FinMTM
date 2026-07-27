# MCP interface contract

Agent inference connects to an SSE MCP endpoint configured by `--mcp-url` or
`FINMTM_MCP_URL`. The default development endpoint is
`http://localhost:8081/sse`.

Every tool call uses:

```json
{
  "name": "<tool name>",
  "arguments": {
    "query": "<natural-language query containing core parameters>"
  }
}
```

The server returns standard MCP content. Text results are appended to the next
ReAct round as tool evidence.

The fixed tool set is:

| Tool | Function |
|---|---|
| `FinQuery` | prices, valuation metrics, trading statistics, fundamentals |
| `StockNews` | real-time stock news retrieval |
| `AnalysisLib` | structured financial analysis |
| `NoticeSearch` | corporate announcements, filings, and disclosures |
| `VisitWeb` | webpage content parsing |

The repository contains the client and scoring pipeline. Deploying or licensing
the underlying financial data services is environment-specific; evaluation of
previously saved traces does not require those services.
