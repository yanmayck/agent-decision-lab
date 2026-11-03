"""Unit tests for TreasureGraph Decision Theory module."""

from __future__ import annotations

import unittest

from treasuregraph.decision_theory import (
    expected_cost_open_first,
    expected_cost_report_first,
)


class TestDecisionTheory(unittest.TestCase):
    """Test suite for expected cost calculations and validation."""

    def test_specified_benchmark_values(self):
        """Verifies requirement: for p=0.9, open_cost=1, report_cost=10, results are 2.1 and 11.0."""
        p = 0.9
        c_open = 1
        c_report = 10

        cost_open = expected_cost_open_first(p, c_open, c_report)
        cost_report = expected_cost_report_first(c_open, c_report)

        self.assertAlmostEqual(cost_open, 2.1, places=6)
        self.assertAlmostEqual(cost_report, 11.0, places=6)

    def test_edge_cases_probability(self):
        """Verifies boundary conditions for probability (p=1.0 and p=0.0)."""
        # When p=1.0, candidate safe is guaranteed to hold the treasure
        self.assertAlmostEqual(expected_cost_open_first(1.0, open_cost=5.0, report_cost=20.0), 5.0)

        # When p=0.0, candidate safe is guaranteed empty (opens safe, then buys report, then opens correct safe)
        # Cost = 5 + (5 + 20) = 30.0
        self.assertAlmostEqual(expected_cost_open_first(0.0, open_cost=5.0, report_cost=20.0), 30.0)

        # Report first is independent of probability of candidate safe
        self.assertAlmostEqual(expected_cost_report_first(open_cost=5.0, report_cost=20.0), 25.0)

    def test_parameter_validations_probability(self):
        """Verifies that invalid probabilities raise appropriate errors."""
        # Out of bounds
        with self.assertRaises(ValueError):
            expected_cost_open_first(-0.1, 1, 10)

        with self.assertRaises(ValueError):
            expected_cost_open_first(1.01, 1, 10)

        # Wrong type
        with self.assertRaises(TypeError):
            expected_cost_open_first("0.9", 1, 10)  # type: ignore

        with self.assertRaises(TypeError):
            expected_cost_open_first(True, 1, 10)  # type: ignore

    def test_parameter_validations_costs(self):
        """Verifies that negative costs or invalid types raise appropriate errors."""
        with self.assertRaises(ValueError):
            expected_cost_open_first(0.5, open_cost=-1, report_cost=10)

        with self.assertRaises(ValueError):
            expected_cost_open_first(0.5, open_cost=1, report_cost=-5)

        with self.assertRaises(ValueError):
            expected_cost_report_first(open_cost=-1, report_cost=10)

        with self.assertRaises(ValueError):
            expected_cost_report_first(open_cost=1, report_cost=-10)

        # Type checks
        with self.assertRaises(TypeError):
            expected_cost_report_first(True, 10)  # type: ignore

        with self.assertRaises(TypeError):
            expected_cost_report_first(1, "10")  # type: ignore


if __name__ == "__main__":
    unittest.main()
