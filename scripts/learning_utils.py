"""Shared behavioral-learning utilities."""

from __future__ import annotations

import numpy as np


def exp_learning(t, a, b, k):
    """Exponential learning curve ``P(t) = a - b * exp(-k*t)``."""
    return a - b * np.exp(-k * t)
