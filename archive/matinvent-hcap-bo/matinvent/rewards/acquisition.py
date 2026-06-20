"""Acquisition functions for calculator routing.

Implements Expected Improvement per Cost for intelligent calculator selection
based on GP uncertainty and computational cost.
"""

import numpy as np
from typing import Dict, List
from scipy.stats import norm


class ExpectedImprovementPerCost:
    """
    Expected Improvement per Cost acquisition function.

    Routes structures to calculators based on:
    - GP uncertainty (higher uncertainty → more valuable to query)
    - Calculator cost (prefer cheaper unless uncertainty is high)
    - Expected improvement in reward

    Cost model:
        - ORB: 0.001 (very cheap, fast MLIP)
        - ALIGNN: 0.01 (cheap ML predictor)
        - VASP: 1.0 (expensive DFT calculation)
    """

    def __init__(
        self,
        cost_model: Dict[str, float],
        xi: float = 0.01
    ):
        """
        Initialize acquisition function.

        Args:
            cost_model: Dictionary mapping calculator names to normalized costs
                        e.g., {'orb': 0.001, 'alignn': 0.01, 'vasp': 1.0}
            xi: Exploration parameter (trade-off between exploitation and exploration)
        """
        self.cost_model = cost_model
        self.xi = xi

    def compute(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        best_observed: float
    ) -> Dict[str, np.ndarray]:
        """
        Compute Expected Improvement per Cost for each calculator.

        Args:
            mean: GP predicted mean values (n_structures,)
            std: GP predicted std (uncertainty) (n_structures,)
            best_observed: Best observed property value so far

        Returns:
            dict: {calculator_name: ei_per_cost_array}
        """
        results = {}

        for calc_name, cost in self.cost_model.items():
            # Expected Improvement (EI)
            # EI = E[max(f(x) - f_best, 0)] where f(x) ~ N(mean, std²)

            # Standardized improvement
            with np.errstate(divide='ignore', invalid='ignore'):
                z = (mean - best_observed - self.xi) / (std + 1e-9)

            # Expected improvement formula
            ei = (mean - best_observed - self.xi) * norm.cdf(z) + std * norm.pdf(z)

            # Handle cases where std = 0 (no uncertainty)
            ei[std == 0.0] = 0.0

            # EI per unit cost
            if cost < 1e-9:
                # Effectively free calculator
                ei_per_cost = np.full_like(ei, np.inf)
            else:
                ei_per_cost = ei / cost

            results[calc_name] = ei_per_cost

        return results

    def select_calculator(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        best_observed: float,
        available_calculators: List[str] = None
    ) -> List[str]:
        """
        Select best calculator for each structure.

        Args:
            mean: GP predicted mean values (n_structures,)
            std: GP predicted std (uncertainty) (n_structures,)
            best_observed: Best observed property value so far
            available_calculators: List of available calculator names
                                   If None, use all in cost_model

        Returns:
            list: Calculator names (one per structure)
        """
        # Default to all calculators in cost model
        if available_calculators is None:
            available_calculators = list(self.cost_model.keys())

        # Filter to only available calculators
        available_costs = {
            name: cost for name, cost in self.cost_model.items()
            if name in available_calculators
        }

        # Compute EI/cost for available calculators
        ei_per_cost_dict = {}
        for calc_name in available_calculators:
            if calc_name in self.cost_model:
                cost = self.cost_model[calc_name]

                # Expected Improvement
                with np.errstate(divide='ignore', invalid='ignore'):
                    z = (mean - best_observed - self.xi) / (std + 1e-9)
                ei = (mean - best_observed - self.xi) * norm.cdf(z) + std * norm.pdf(z)
                ei[std == 0.0] = 0.0

                # EI per cost
                if cost < 1e-9:
                    ei_per_cost = np.full_like(ei, np.inf)
                else:
                    ei_per_cost = ei / cost

                ei_per_cost_dict[calc_name] = ei_per_cost

        # Stack into array: (n_structures, n_calculators)
        calc_names = list(ei_per_cost_dict.keys())
        ei_per_cost_array = np.stack([
            ei_per_cost_dict[name] for name in calc_names
        ], axis=1)

        # Select calculator with highest EI/cost for each structure
        best_calc_idx = np.argmax(ei_per_cost_array, axis=1)
        selected_calcs = [calc_names[idx] for idx in best_calc_idx]

        return selected_calcs


class UncertaintyThresholdRouting:
    """
    Simple threshold-based routing (alternative to EI).

    Routes based on fixed uncertainty thresholds:
    - Low uncertainty → cheap calculator
    - High uncertainty → expensive calculator
    """

    def __init__(
        self,
        thresholds: Dict[str, float],
        calculators: List[str]
    ):
        """
        Initialize threshold router.

        Args:
            thresholds: {calculator: max_std_threshold}
            calculators: Ordered list from cheapest to most expensive
        """
        self.thresholds = thresholds
        self.calculators = calculators

    def select_calculator(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        **kwargs
    ) -> List[str]:
        """
        Select calculator based on uncertainty thresholds.

        Args:
            mean: GP predicted mean (unused, for API compatibility)
            std: GP predicted uncertainty

        Returns:
            list: Calculator names (one per structure)
        """
        selected = []

        for uncertainty in std:
            # Start from most expensive, find first where uncertainty exceeds threshold
            calc = self.calculators[0]  # Default to cheapest

            for calc_name in reversed(self.calculators):
                threshold = self.thresholds.get(calc_name, 0)
                if uncertainty > threshold:
                    calc = calc_name
                    break

            selected.append(calc)

        return selected
