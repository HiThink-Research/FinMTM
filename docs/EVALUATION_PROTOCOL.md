# Paper-aligned evaluation protocol

This document records the scoring logic implemented by the public repository.
The submitted manuscript is the source of truth.

## Objective questions

For predicted option set `P` and gold set `G`:

```text
score = 0                         if P contains any option outside G
score = |P intersection G| / |G| otherwise
```

Scores are in 0-1 and are multiplied by 100 for reporting. This gives partial
credit only when the model does not over-select.

## Open-ended dialogues

Each response is scored from 0 to 10 on:

- visual precision (VP)
- financial logic (FL)
- data accuracy (DA)
- cross-modal verification (CMV)
- temporal awareness (TA)

All five capability weights are 0.2. Scores are then averaged over turns. The
session-level judge uses the session's task label and corresponding checklist:

- L1 / Comprehension: entity recognition and spatial awareness
- L2 / Calculation: multi-step calculation and chart estimation
- L3 / Self-correction: adversarial robustness and logical consistency
- L4 / Memory: cross-page linking, long context, and multi-source fusion

The final score is `0.5 * turn_score + 0.5 * session_score`. Internal scores are
0-10; output files explicitly report the benchmark scale of 0-100.

## Financial Agent

Tool calls are represented as `(name/function, core arguments)` and matched
one-to-one as unordered sets. Matching is semantic and task-functional; order
and exact strings are not evaluation targets.

```text
precision = TP / predicted_count
recall    = TP / reference_count
F2        = 5 * precision * recall / (4 * precision + recall)
EMR       = 1 if predicted and reference calls match exactly, else 0
Qt        = 25 * F2
Qfinal    = Qa + Qr + Qt
```

`Qa` is a graded 0-50 answer score. `Qr` is a 0-25 evidence-grounded reasoning
score. `Qt` is a 0-25 planning score. `EMR` is reported only as a diagnostic and
does not enter `Qfinal`. All arithmetic is performed in `finmtm_eval/metrics.py`.
