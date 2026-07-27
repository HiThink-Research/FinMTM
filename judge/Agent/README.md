# Financial Agent evaluation

This pipeline implements manuscript Equations (5)-(6).

## Scoring

- `answer_score` (`Qa`): graded 0-50. Semantically equivalent numerical and
  formatting expressions are not penalized.
- `reasoning_score` (`Qr`): 0-25, based on coherence and grounding in visual
  evidence and actual tool returns.
- `tool_score` (`Qt`): `25 * F2` with `beta=2`, where semantic matching is
  one-to-one and
  based on tool function plus core arguments. Call order is ignored.
- `emr`: exact-match diagnostic in `{0, 1}`; it is reported but does not enter
  the total score.
- `total_score`: `Qa + Qr + Qt`, from 0 to 100.

The LLM judge reports indexed semantic match pairs and the two rubric scores.
`judge.py` validates one-to-one match indices and recomputes precision, recall,
F2, EMR, the tool score, and total score in Python. Judge-provided arithmetic is
not trusted. Inference traces retain structured tool results and accumulated
tool feedback so answer and reasoning scores can be audited against actual
evidence.

## Fixed MCP tool set

`FinQuery`, `StockNews`, `AnalysisLib`, `NoticeSearch`, and `VisitWeb`.
See [`../../docs/MCP_INTERFACE.md`](../../docs/MCP_INTERFACE.md) for the public
transport contract. The financial data service itself is external to this
evaluation repository.

## Commands

Run inference and evaluation:

```bash
python -m judge.Agent.main \
  --mode all \
  --input /path/to/agent.jsonl \
  --out-root outputs/agent \
  --api-base http://localhost:8000/v1 \
  --model Qwen3-VL-30B-A3B-Instruct \
  --judge-api-base http://localhost:8000/v1 \
  --judge-model ChatGPT-4o \
  --mcp-url http://localhost:8081/sse
```

Evaluate an existing trace:

```bash
python -m judge.Agent.main \
  --mode eval \
  --input /path/to/trace.jsonl \
  --out-root outputs/agent_scores \
  --judge-api-base http://localhost:8000/v1 \
  --judge-model ChatGPT-4o
```

`--mode eval` does not require an MCP server.
