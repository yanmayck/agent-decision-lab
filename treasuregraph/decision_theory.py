"""Decision Theory utilities for TreasureGraph.

Calculates expected costs and optimal decision thresholds under uncertainty,
comparing immediate action (open first) versus information gathering (report first).
"""

from __future__ import annotations


def _validate_numeric_cost(name: str, value: float) -> float:
    """Validates that a cost parameter is a non-negative real number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"'{name}' must be a real number, got {type(value).__name__}.")
    if value < 0:
        raise ValueError(f"'{name}' must be non-negative, got {value}.")
    return float(value)


def _validate_probability(name: str, value: float) -> float:
    """Validates that a probability parameter is within [0.0, 1.0]."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"'{name}' must be a real number, got {type(value).__name__}.")
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"'{name}' must be between 0.0 and 1.0, got {value}.")
    return float(value)


def expected_cost_open_first(
    probability: float,
    open_cost: float,
    report_cost: float,
) -> float:
    """Calculates expected cost of opening the most probable safe first.

    Strategy:
    - Open the candidate safe (incurs open_cost).
    - With probability `p`, the treasure is found. Total cost = open_cost.
    - With probability `1 - p`, the safe is empty. The agent must then
      consult the perfect report (incurs report_cost) and open the confirmed
      correct safe (incurs open_cost). Total cost in this branch = 2 * open_cost + report_cost.

    Expected cost:
        E = p * open_cost + (1 - p) * (2 * open_cost + report_cost)
          = open_cost + (1 - p) * (open_cost + report_cost)

    Args:
        probability: Probability p in [0.0, 1.0] that candidate safe has the treasure.
        open_cost: Non-negative cost of opening a safe.
        report_cost: Non-negative cost of obtaining the perfect report.

    Returns:
        float expected cost.
    """
    p = _validate_probability("probability", probability)
    c_open = _validate_numeric_cost("open_cost", open_cost)
    c_report = _validate_numeric_cost("report_cost", report_cost)

    return round(c_open + (1.0 - p) * (c_open + c_report), 10)


def expected_cost_report_first(
    open_cost: float,
    report_cost: float,
) -> float:
    """Calculates expected cost of consulting the perfect report first.

    Strategy:
    - Consult the perfect report with certainty (incurs report_cost).
    - Open the designated correct safe directly (incurs open_cost).

    Expected cost:
        E = report_cost + open_cost

    Args:
        open_cost: Non-negative cost of opening a safe.
        report_cost: Non-negative cost of obtaining the perfect report.

    Returns:
        float expected cost.
    """
    c_open = _validate_numeric_cost("open_cost", open_cost)
    c_report = _validate_numeric_cost("report_cost", report_cost)

    return round(c_report + c_open, 10)
