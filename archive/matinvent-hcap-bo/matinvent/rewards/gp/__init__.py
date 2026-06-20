"""GP Surrogate module for MatInvent.

Provides Gaussian Process surrogate models using BoTorch for
uncertainty-aware property prediction and calculator routing.
"""

from rewards.gp.surrogate import GPSurrogate
from rewards.gp.metrics import GPMetrics, calculate_metrics

__all__ = ['GPSurrogate', 'GPMetrics', 'calculate_metrics']
