# Evaluation data format

All benchmark inputs and outputs use UTF-8 JSONL: one JSON object per line.
Paths may be absolute, relative to the working directory, or HTTP(S) URLs.

## Objective questions

Each input record follows the chat-style dataset format used by the release:

- `messages[0].content`: question text plus one or more image references
- `choices[0].message.content[0].text`: JSON containing the gold `answer`

The answer is either one option label or a list of labels.

## Open-ended dialogues

Required fields:

- `sample_id` or `session_id`
- `task_type` or `level`: `L1`, `L2`, `L3`, or `L4`
- `image_path` or `image_paths`
- `turns`: ordered list with `turn_id`, `question`, `gold_answer`, and
  `model_answer`

For backward compatibility, the level can be inferred from filenames such as
`L2_with_id_vlm.jsonl`, but an explicit session label is preferred.

## Financial Agent source records

Required fields:

- `sample_id`
- `question` or `turn.question`
- `image_path`
- reference `ActionTrace`, visual observation, thought, and gold conclusion

Agent inference writes `trace.jsonl` with model/reference answers, observations,
thoughts, and tool-call sets. Agent evaluation consumes that trace directly.

Dataset splits are task-based (`objective`, `open-ended L1-L4`, and `agent`).
Fuzz and non-fuzz Agent subsets must be carried as explicit metadata in the
released dataset rather than inferred from filesystem paths.
