import asyncio
import json
import unittest

from judge.Agent.judge import eval_one_sample, resolve_tool_metrics
from judge.Agent.prompts import build_agent_eval_prompt


class AgentJudgeTests(unittest.TestCase):
    def test_tool_counts_drive_deterministic_metrics(self):
        sample = {
            "model_tool_calls": [{"tool": "A"}, {"tool": "B"}, {"tool": "C"}],
            "reference_tool_calls": [{"tool": "A"}, {"tool": "B"}],
        }
        parsed = {"tool_metrics": {"true_positives": 2}}
        metrics = resolve_tool_metrics(parsed, sample)
        self.assertEqual(metrics["precision"], 2 / 3)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["true_positives"], 2)

    def test_prompt_matches_manuscript_protocol(self):
        prompt = build_agent_eval_prompt(
            {"model_tool_feedback": "FinQuery: evidence"}
        )
        self.assertIn("0～50", prompt)
        self.assertIn("F2×25", prompt)
        self.assertIn("β=2", prompt)
        self.assertNotIn("0 或 50", prompt)
        self.assertIn("EMR", prompt)
        self.assertIn("matched_pairs", prompt)
        self.assertIn("FinQuery: evidence", prompt)

    def test_identical_duplicate_calls_are_set_deduplicated(self):
        sample = {
            "model_tool_calls": [{"tool": "A"}, {"tool": "A"}],
            "reference_tool_calls": [{"tool": "A"}],
        }
        metrics = resolve_tool_metrics(
            {"tool_metrics": {"true_positives": 1}},
            sample,
        )
        self.assertEqual(metrics["predicted_count"], 1)
        self.assertEqual(metrics["precision"], 1.0)

    def test_continuous_answer_score_and_f2_total(self):
        class Client:
            async def image2text(self, **_kwargs):
                return json.dumps(
                    {
                        "answer_score": 37.5,
                        "reasoning_score": 18,
                        "tool_metrics": {"true_positives": 1},
                    }
                )

        class Logger:
            def error(self, _message):
                raise AssertionError(_message)

        sample = {
            "model_tool_calls": [{"tool": "A"}, {"tool": "B"}],
            "reference_tool_calls": [{"tool": "A"}],
        }
        result = asyncio.run(eval_one_sample(Client(), sample, Logger()))
        self.assertEqual(result["scores"]["answer"], 37.5)
        self.assertEqual(result["scores"]["reasoning"], 18.0)
        self.assertEqual(result["scores"]["tool"], 20.8333)
        self.assertEqual(result["scores"]["total"], 76.3333)
        self.assertEqual(result["metrics"]["tool_emr"], 0)
        self.assertEqual(result["evaluation_status"], "ok")

    def test_match_pairs_are_validated_one_to_one(self):
        sample = {
            "model_tool_calls": [{"tool": "A"}, {"tool": "B"}],
            "reference_tool_calls": [{"tool": "A"}, {"tool": "B"}],
        }
        parsed = {
            "tool_metrics": {
                "matched_pairs": [
                    {"predicted_index": 0, "reference_index": 0},
                    {"predicted_index": 1, "reference_index": 0},
                    {"predicted_index": 1, "reference_index": 1},
                    {"predicted_index": 99, "reference_index": 1},
                ]
            }
        }
        metrics = resolve_tool_metrics(parsed, sample)
        self.assertEqual(metrics["true_positives"], 2)
        self.assertEqual(len(metrics["matched_pairs"]), 2)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)

    def test_invalid_judge_response_is_flagged(self):
        class Client:
            async def image2text(self, **_kwargs):
                return "not json"

        class Logger:
            def error(self, _message):
                pass

        result = asyncio.run(eval_one_sample(Client(), {}, Logger()))
        self.assertEqual(result["evaluation_status"], "invalid_response")


if __name__ == "__main__":
    unittest.main()
