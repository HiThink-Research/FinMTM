import math
import unittest

from finmtm_eval.metrics import (
    agent_score,
    dialogue_score,
    f_beta_score,
    objective_set_score,
    turn_capability_score,
)


class ObjectiveMetricTests(unittest.TestCase):
    def test_partial_credit_without_overselection(self):
        self.assertEqual(objective_set_score(["A"], ["A", "C"]), 0.5)

    def test_overselection_is_zero(self):
        self.assertEqual(objective_set_score(["A", "B"], ["A", "C"]), 0.0)

    def test_full_match(self):
        self.assertEqual(objective_set_score(["c", "A"], ["A", "C"]), 1.0)


class OpenEndedMetricTests(unittest.TestCase):
    def test_uniform_turn_capability_average(self):
        self.assertEqual(turn_capability_score([10, 8, 6, 4, 2]), 6.0)

    def test_dialogue_alpha_and_reporting_scale(self):
        self.assertEqual(dialogue_score(8, 6), 70.0)


class AgentMetricTests(unittest.TestCase):
    def test_f2_penalises_missing_required_calls(self):
        self.assertTrue(
            math.isclose(f_beta_score(1.0, 0.5), 5 / 9, rel_tol=1e-9)
        )

    def test_f2_tolerates_extra_calls_when_recall_is_complete(self):
        self.assertTrue(
            math.isclose(f_beta_score(0.5, 1.0), 5 / 6, rel_tol=1e-9)
        )

    def test_agent_total_uses_paper_ranges(self):
        result = agent_score(40, 20, 1.0, 0.5)
        self.assertTrue(math.isclose(result["tool"], 125 / 9, rel_tol=1e-9))
        self.assertTrue(math.isclose(result["total"], 665 / 9, rel_tol=1e-9))


if __name__ == "__main__":
    unittest.main()
