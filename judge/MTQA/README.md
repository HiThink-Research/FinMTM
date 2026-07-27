# Open-ended multi-turn evaluation

The evaluator implements manuscript Equations (2)-(4).

1. Every turn receives five 0-10 scores: visual precision, financial logic,
   data accuracy, cross-modal verification, and temporal awareness.
2. The five capabilities and all turns are uniformly averaged to obtain a
   turn-level score on 0-10.
3. Each session is evaluated with its explicit L1-L4 checklist on 0-10.
4. The final internal score is `0.5 * turn + 0.5 * session`.
5. Outputs are multiplied by ten for the 0-100 reporting scale used in the
   benchmark tables.

There are no level-specific 0.4/0.6 weights, citation gates, robustness gates,
or multiplicative penalties in the paper-aligned implementation.

Run from the repository root:

```bash
python -m judge.MTQA.main \
  --dirs /path/to/model_outputs \
  --pattern "L*_with_id_vlm.jsonl" \
  --out_subdir scores \
  --client openai \
  --api_base http://localhost:8000/v1 \
  --model ChatGPT-4o
```
