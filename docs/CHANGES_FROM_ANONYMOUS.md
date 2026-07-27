# Changes from the early anonymized snapshot

The anonymized repository snapshot predated the evaluation protocol described
in the submitted manuscript. This revision treats the manuscript as the source
of truth.

| Area | Early snapshot | Paper-aligned revision |
|---|---|---|
| Objective multi-choice | exact set equality | no-overselection plus proportional partial credit |
| Turn score | judge-provided overall | deterministic uniform mean of VP/FL/DA/CMV/TA |
| Open-ended aggregation | mixed 0.4/0.6 and 0.5/0.5 | fixed 0.5/0.5 for every task |
| Open-ended gates | citation/robustness multiplicative gates | removed; not present in Equation (4) |
| Session checklist | extra implementation-specific items | exact L1-L4 items listed in manuscript Table 2 |
| Turn judge range | prompt floor of 1 | full manuscript range of 0-10 |
| Open-ended scale | mixed 0-10 and 0-100 | internal 0-10, explicit 0-100 reporting |
| L1 session judge | missing function | explicit L1-L4 task checklists |
| Agent answer | binary 0/50 prompt | graded 0-50 rubric |
| Agent tool score | F1 × 25 | F2 × 25 with beta=2 |
| Agent trajectory | sequence EMR diagnostic | unordered one-to-one functional matching; EMR retained as a diagnostic |
| Agent evidence | tool returns used online but omitted from saved trace | structured tool results and accumulated feedback retained for judging |
| Agent arithmetic | accepted judge totals | deterministic Python aggregation |
| Tool set | old Search/ReportQuery names | fixed five-tool set from Table 11 |
| Runtime | hard-coded internal paths and `api2` | command-line paths and public OpenAI-compatible adapter |
| MCP | implicit local dependency | documented SSE client contract and offline trace evaluation |

The repository tests validate the formulas and aggregation code. They do not
reconstruct or claim new benchmark results; model outputs must be rerun to
measure any effect on reported tables.
