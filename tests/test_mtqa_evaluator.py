import asyncio
import json
import unittest

from judge.MTQA.evaluator import evaluate_sample


class FakeJudgeClient:
    def chat(self, image=None, text=None):
        if "Checklist_Scores" in text:
            return json.dumps(
                {
                    "Checklist_Scores": {
                        "Multi_Step_Numerical_Calculation": 6,
                        "Chart_Numerical_Estimation": 6,
                    },
                    "Pass": True,
                    "Critique": "ok",
                }
            )
        return json.dumps(
            {
                "Visual_Precision": 8,
                "Financial_Logic": 8,
                "Data_Accuracy": 8,
                "Cross_Modal_Verification": 8,
                "Temporal_Awareness": 8,
                "Comment": "ok",
            }
        )


class OpenEndedEvaluatorTests(unittest.TestCase):
    def test_uniform_aggregation_and_scale(self):
        sample = {
            "sample_id": "x",
            "task_type": "L2",
            "turns": [
                {
                    "turn_id": "T1",
                    "question": "q",
                    "gold_answer": "g",
                    "model_answer": "p",
                }
            ],
        }
        result = asyncio.run(evaluate_sample(sample, FakeJudgeClient()))
        self.assertEqual(result["avg_turn_score_0_10"], 8.0)
        self.assertEqual(result["session_score_0_10"], 6.0)
        self.assertEqual(result["final_composite_score"], 70.0)
        self.assertEqual(result["evaluation_status"], "ok")
        self.assertEqual(
            result["capability_scores_0_100"]["visual_precision"],
            80.0,
        )

    def test_empty_dialogue_is_rejected(self):
        sample = {"sample_id": "empty", "task_type": "L1", "turns": []}
        with self.assertRaises(ValueError):
            asyncio.run(evaluate_sample(sample, FakeJudgeClient()))

    def test_invalid_judge_response_is_flagged(self):
        class InvalidJudgeClient:
            def chat(self, image=None, text=None):
                return "{}"

        sample = {
            "sample_id": "invalid",
            "task_type": "L1",
            "turns": [
                {
                    "turn_id": "T1",
                    "question": "q",
                    "gold_answer": "g",
                    "model_answer": "p",
                }
            ],
        }
        result = asyncio.run(evaluate_sample(sample, InvalidJudgeClient()))
        self.assertEqual(result["evaluation_status"], "error")


if __name__ == "__main__":
    unittest.main()
