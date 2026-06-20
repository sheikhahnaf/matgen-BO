"""Noise estimator for learning calculator noise levels from paired data.

Learns heteroscedastic noise levels by comparing calculator outputs to
ground truth (highest fidelity calculator available).
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from memory.ltm import LongTimeMem


class NoiseEstimator:
    """
    Learn calculator noise levels from paired data.

    Estimates noise by computing standard deviation of errors relative to
    ground truth calculator (highest fidelity available).

    Workflow:
        1. Collect paired data during calibration phase (all calculators on same structures)
        2. Compute errors: error_i = calc_i(x) - ground_truth(x)
        3. Noise_i = std(error_i)
        4. Use noise levels in heteroscedastic GP
    """

    def __init__(
        self,
        calculator_hierarchy: List[str] = None,
        min_paired_samples: int = 20,
        correct_systematic_bias: bool = False,
        noise_floor: float = 0.1
    ):
        """
        Initialize noise estimator.

        Args:
            calculator_hierarchy: Ordered list of calculators by fidelity (highest first)
                                 e.g., ['vasp', 'orb', 'alignn']
                                 Ground truth = first available calculator
            min_paired_samples: Minimum paired samples needed for estimation
            correct_systematic_bias: If True, remove mean bias before computing noise
            noise_floor: Minimum noise level (avoid zero noise)
        """
        self.calculator_hierarchy = calculator_hierarchy or ['vasp', 'orb', 'alignn']
        self.min_paired_samples = min_paired_samples
        self.correct_systematic_bias = correct_systematic_bias
        self.noise_floor = noise_floor

        self.noise_estimates = None
        self.bias_estimates = None
        self.ground_truth_calculator = None
        self.n_paired_samples = 0

    def estimate_from_ltm(self, ltm: LongTimeMem) -> Optional[Dict[str, float]]:
        """
        Estimate noise levels from paired calculator data in LTM.

        Args:
            ltm: LongTimeMem instance with multi-calculator data

        Returns:
            dict: {calculator_name: noise_level} or None if insufficient data
        """
        # Extract paired data
        paired_data = self._extract_paired_data(ltm)

        if paired_data is None or len(paired_data) < self.min_paired_samples:
            logging.warning(
                f"Insufficient paired samples for noise estimation: "
                f"{len(paired_data) if paired_data is not None else 0} < {self.min_paired_samples}"
            )
            return None

        self.n_paired_samples = len(paired_data)

        # Determine ground truth calculator (highest fidelity available)
        available_calcs = set(paired_data.columns)
        ground_truth = None
        for calc in self.calculator_hierarchy:
            if calc in available_calcs:
                ground_truth = calc
                break

        if ground_truth is None:
            logging.error("No valid ground truth calculator found in paired data")
            return None

        self.ground_truth_calculator = ground_truth
        logging.info(f"Using '{ground_truth}' as ground truth for noise estimation")

        # Compute noise levels
        noise_levels = {ground_truth: self.noise_floor}  # Ground truth gets minimal noise
        bias_levels = {ground_truth: 0.0}

        gt_values = paired_data[ground_truth].values

        for calc in available_calcs:
            if calc == ground_truth:
                continue

            calc_values = paired_data[calc].values

            # Compute errors
            errors = calc_values - gt_values

            # Estimate systematic bias
            bias = np.mean(errors)
            bias_levels[calc] = bias

            # Estimate noise (random error)
            if self.correct_systematic_bias:
                # Remove bias before computing noise
                noise = np.std(errors - bias)
            else:
                # Include bias in noise estimate (more conservative)
                noise = np.std(errors)

            noise_levels[calc] = max(noise, self.noise_floor)

        self.noise_estimates = noise_levels
        self.bias_estimates = bias_levels

        # Log results
        logging.info(f"Noise estimation complete from {self.n_paired_samples} paired samples")
        logging.info(f"Ground truth: {ground_truth}")
        logging.info(f"Noise levels: {self._format_dict(noise_levels)}")
        if self.correct_systematic_bias:
            logging.info(f"Systematic biases: {self._format_dict(bias_levels)}")

        return noise_levels

    def _extract_paired_data(self, ltm: LongTimeMem) -> Optional[pd.DataFrame]:
        """
        Extract structures that have been evaluated by multiple calculators.

        Args:
            ltm: LongTimeMem instance

        Returns:
            DataFrame with columns [structure_id, calc1_value, calc2_value, ...]
            or None if insufficient data
        """
        if 'structure_id' not in ltm.memory.columns:
            logging.error("LTM missing 'structure_id' column - cannot extract paired data")
            return None

        if 'calculator_used' not in ltm.memory.columns or 'property_value' not in ltm.memory.columns:
            logging.error("LTM missing required columns for noise estimation")
            return None

        # Filter out null property values
        valid_data = ltm.memory[ltm.memory['property_value'].notna()].copy()

        if len(valid_data) == 0:
            return None

        # Pivot to get structure_id × calculator matrix
        try:
            paired = valid_data.pivot_table(
                index='structure_id',
                columns='calculator_used',
                values='property_value',
                aggfunc='first'  # Take first value if duplicates
            )
        except Exception as e:
            logging.error(f"Failed to pivot LTM data: {e}")
            return None

        # Keep only structures with at least 2 calculator evaluations
        paired = paired.dropna(thresh=2)

        logging.info(f"Extracted {len(paired)} structures with multi-calculator evaluations")

        return paired

    def get_noise_variance(self, calculator: str) -> float:
        """
        Get noise variance (σ²) for a calculator.

        Args:
            calculator: Calculator name

        Returns:
            float: Noise variance (squared noise level)
        """
        if self.noise_estimates is None:
            # Default noise if not yet estimated
            default_noise = 10.0
            return default_noise ** 2

        noise = self.noise_estimates.get(calculator, 10.0)
        return noise ** 2

    def get_noise_array(self, calculators: List[str]) -> np.ndarray:
        """
        Get array of noise variances for a list of calculators.

        Args:
            calculators: List of calculator names

        Returns:
            np.ndarray: Noise variances (σ²) for each calculator
        """
        return np.array([self.get_noise_variance(calc) for calc in calculators])

    def _format_dict(self, d: Dict[str, float]) -> str:
        """Format dict for logging."""
        return ", ".join([f"{k}: {v:.3f}" for k, v in d.items()])

    def save_estimates(self, save_path: str):
        """Save noise estimates to file."""
        if self.noise_estimates is None:
            logging.warning("No noise estimates to save")
            return

        df = pd.DataFrame({
            'calculator': list(self.noise_estimates.keys()),
            'noise_level': list(self.noise_estimates.values()),
            'bias': [self.bias_estimates.get(c, 0.0) for c in self.noise_estimates.keys()]
        })

        df['ground_truth'] = df['calculator'] == self.ground_truth_calculator
        df['n_paired_samples'] = self.n_paired_samples

        df.to_csv(save_path, index=False)
        logging.info(f"Saved noise estimates to {save_path}")
