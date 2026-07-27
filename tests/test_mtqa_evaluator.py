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
                        "Unit_Consistency": 6,
                        "Cross_Turn_Logic": 6,
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
        self.assertEqual(
            result["capability_scores_0_100"]["visual_precision"],
            80.0,
        )


if __name__ == "__main__":
    unittest.main()
