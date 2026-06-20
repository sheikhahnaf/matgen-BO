"""Calculator Router for GP-based intelligent routing.

Routes structures to appropriate calculators based on GP uncertainty
and acquisition function (Expected Improvement per Cost).
"""

import os
import logging
import numpy as np
from typing import Dict, List, Tuple, Any
from pymatgen.core.structure import Structure

from rewards.calculators.base import Calculator
from rewards.gp.surrogate import GPSurrogate
from rewards.calculators.orb.featurizer import ORBFeaturizer
from rewards.acquisition import ExpectedImprovementPerCost


class CalculatorRouter:
    """
    Intelligent calculator router using GP surrog

ates.

    Routes structures to calculators based on:
    1. GP uncertainty estimates
    2. Acquisition function (EI per cost)
    3. Computational cost considerations

    Workflow:
        1. Extract ORB features from structures
        2. Predict property + uncertainty with GP
        3. Compute acquisition function
        4. Route each structure to best calculator
        5. Execute calculations in batches
        6. Return properties + metadata
    """

    def __init__(
        self,
        calculators: Dict[str, Calculator],
        gp_model: GPSurrogate,
        featurizer: ORBFeaturizer,
        acquisition_fn: ExpectedImprovementPerCost,
        default_calculator: str = 'orb',
        min_gp_samples: int = 10,
        calibration_mode: bool = False,
        uncertainty_threshold: float = None  # NEW: If set, only query calculator when uncertainty > threshold
    ):
        """
        Initialize Calculator Router.

        Args:
            calculators: Dictionary {name: Calculator instance}
            gp_model: GPSurrogate instance
            featurizer: ORBFeaturizer instance
            acquisition_fn: Acquisition function for routing
            default_calculator: Calculator to use before GP is trained
            min_gp_samples: Minimum samples before enabling GP routing
            calibration_mode: If True, query all calculators (for noise estimation)
            uncertainty_threshold: If set, only query calculator when GP uncertainty > threshold.
                                  Otherwise use GP mean directly (cost-effective with single calculator)
        """
        self.calculators = calculators
        self.gp_model = gp_model
        self.featurizer = featurizer
        self.acquisition_fn = acquisition_fn
        self.default_calculator = default_calculator
        self.min_gp_samples = min_gp_samples
        self.calibration_mode = calibration_mode
        self.uncertainty_threshold = uncertainty_threshold

    def calibration_compute(
        self,
        samples: Tuple[List[Structure], str],
        label: str = 'tmp'
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Query ALL calculators during calibration phase (for noise estimation).

        Args:
            samples: Tuple of (structures, xyz_path)
            label: Label for saving results

        Returns:
            multi_calc_properties: Dict {calc_name: np.ndarray of properties}
            metadata: Dict with cost and mode info
        """
        structures, xyz_path = samples
        n_structures = len(structures)

        logging.info(f"Router (CALIBRATION): Querying all {len(self.calculators)} calculators on {n_structures} structures")

        multi_calc_properties = {}
        total_cost = 0.0

        # Query each calculator
        for calc_name, calc in self.calculators.items():
            logging.info(f"Router (CALIBRATION): Running {calc_name}...")
            try:
                properties = calc.calc(samples, f"{label}_{calc_name}")
                multi_calc_properties[calc_name] = properties

                # Track cost
                cost = n_structures * self.acquisition_fn.cost_model[calc_name]
                total_cost += cost
            except Exception as e:
                logging.error(f"Router (CALIBRATION): {calc_name} failed: {str(e)}")
                multi_calc_properties[calc_name] = np.full(n_structures, np.nan)

        metadata = {
            'mode': 'calibration',
            'cost': total_cost,
            'n_calculators': len(self.calculators),
            'calculator_names': list(self.calculators.keys())
        }

        logging.info(f"Router (CALIBRATION): Total cost: {total_cost:.4f}")

        return multi_calc_properties, metadata

    def route_and_compute(
        self,
        samples: Tuple[List[Structure], str],
        label: str = 'tmp'
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Route structures to calculators and compute properties.

        If calibration_mode=True, queries ALL calculators.
        Otherwise, uses GP-based intelligent routing.

        Args:
            samples: Tuple of (structures, xyz_path)
            label: Label for saving results

        Returns:
            properties: np.ndarray of computed properties (or dict if calibration)
            metadata: Dict with routing info, uncertainties, costs, etc.
        """
        # Calibration mode: query all calculators
        if self.calibration_mode:
            return self.calibration_compute(samples, label)

        # Normal mode: GP-based routing
        structures, xyz_path = samples
        n_structures = len(structures)

        logging.info(f"Router: Processing {n_structures} structures")

        # Check if GP has enough training data
        if not self.gp_model.is_trained or \
           self.gp_model.get_training_data_size() < self.min_gp_samples:
            logging.info(f"Router: GP not ready (need {self.min_gp_samples} samples), using default calculator")
            return self._route_default(samples, label)

        # Extract ORB features
        logging.info("Router: Extracting ORB features...")
        features = self.featurizer.featurize(structures)

        # GP prediction
        logging.info("Router: GP prediction...")
        mean, std = self.gp_model.predict(features, return_std=True)

        # Get best observed value for acquisition function
        best_observed = self._get_best_observed()

        # NEW: Cost-effective routing with uncertainty threshold
        # When uncertainty < threshold, use GP predictions instead of querying calculator
        if self.uncertainty_threshold is not None:
            logging.info(f"Router: Using uncertainty-based routing (threshold={self.uncertainty_threshold:.4f})")

            # Classify samples: high uncertainty (query calculator) vs low uncertainty (use GP)
            std_flat = std.flatten()
            high_uncertainty_mask = std_flat > self.uncertainty_threshold
            low_uncertainty_mask = ~high_uncertainty_mask

            n_high = high_uncertainty_mask.sum()
            n_low = low_uncertainty_mask.sum()

            logging.info(f"Router: High uncertainty (query calculator): {n_high} samples ({100*n_high/n_structures:.1f}%)")
            logging.info(f"Router: Low uncertainty (use GP mean): {n_low} samples ({100*n_low/n_structures:.1f}%)")

            # Initialize properties with GP predictions
            properties = mean.flatten().copy()
            total_cost = 0.0

            # Track which samples used GP vs calculator
            source_tracker = np.where(low_uncertainty_mask, 'gp_prediction', 'calculator')

            # Only query calculator for high uncertainty samples
            if n_high > 0:
                # Use default calculator for high uncertainty samples
                calc_name = self.default_calculator
                calc = self.calculators[calc_name]

                # Get indices of high uncertainty samples
                high_uncertainty_indices = np.where(high_uncertainty_mask)[0]

                logging.info(f"Router: Querying {calc_name} for {n_high} high-uncertainty samples...")

                # Prepare batch
                batch_structures = [structures[i] for i in high_uncertainty_indices]
                batch_samples = (batch_structures, xyz_path)
                batch_label = f"{label}_{calc_name}_highuncert"

                # Execute calculator
                try:
                    batch_properties = calc.calc(batch_samples, batch_label)
                    properties[high_uncertainty_indices] = batch_properties

                    # Track cost (only for calculator queries)
                    cost = n_high * self.acquisition_fn.cost_model[calc_name]
                    total_cost += cost

                except Exception as e:
                    logging.error(f"Router: {calc_name} failed: {str(e)}")
                    # Keep GP predictions for failed samples

            # Metadata
            metadata = {
                'routed_to': source_tracker.tolist(),  # 'gp_prediction' or 'calculator'
                'uncertainties': std_flat,
                'gp_predictions': mean.flatten(),
                'features': features,
                'cost': total_cost,
                'routing_counts': {
                    'gp_prediction': n_low,
                    self.default_calculator: n_high
                },
                'best_observed': best_observed,
                'uncertainty_threshold': self.uncertainty_threshold,
                'cost_savings': n_low * self.acquisition_fn.cost_model[self.default_calculator],  # Cost saved by using GP
            }

            logging.info(f"Router: Total cost: {total_cost:.4f} (saved {metadata['cost_savings']:.4f} by using GP)")

            return properties, metadata

        # ORIGINAL: Multi-calculator routing based on acquisition function
        else:
            # Route based on acquisition function
            logging.info("Router: Computing acquisition function...")
            available_calcs = list(self.calculators.keys())
            selected_calcs = self.acquisition_fn.select_calculator(
                mean.flatten(),
                std.flatten(),
                best_observed,
                available_calculators=available_calcs
            )

            # Group structures by calculator
            calc_groups = {}
            for i, calc_name in enumerate(selected_calcs):
                if calc_name not in calc_groups:
                    calc_groups[calc_name] = []
                calc_groups[calc_name].append(i)

            logging.info(f"Router: Routing breakdown: {self._format_routing_counts(calc_groups, n_structures)}")

            # Compute properties for each group
            properties = np.full(n_structures, np.nan)
            total_cost = 0.0

            for calc_name, indices in calc_groups.items():
                calc = self.calculators[calc_name]
                n_batch = len(indices)

                logging.info(f"Router: Running {calc_name} on {n_batch} structures...")

                # Prepare batch
                batch_structures = [structures[i] for i in indices]
                batch_samples = (batch_structures, xyz_path)
                batch_label = f"{label}_{calc_name}"

                # Execute calculator
                try:
                    batch_properties = calc.calc(batch_samples, batch_label)
                    properties[indices] = batch_properties

                    # Track cost
                    cost = n_batch * self.acquisition_fn.cost_model[calc_name]
                    total_cost += cost

                except Exception as e:
                    logging.error(f"Router: {calc_name} failed: {str(e)}")
                    # Leave as NaN

            # Metadata
            metadata = {
                'routed_to': selected_calcs,
                'uncertainties': std.flatten(),
                'gp_predictions': mean.flatten(),
                'features': features,
                'cost': total_cost,
                'routing_counts': {k: len(v) for k, v in calc_groups.items()},
                'best_observed': best_observed,
            }

            return properties, metadata

    def _route_default(
        self,
        samples: Tuple[List[Structure], str],
        label: str
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Route all structures to default calculator (cold start).

        Args:
            samples: Tuple of (structures, xyz_path)
            label: Label for saving results

        Returns:
            properties: np.ndarray of computed properties
            metadata: Dict with routing info
        """
        structures, _ = samples
        n_structures = len(structures)

        # Use default calculator for all
        calc = self.calculators[self.default_calculator]
        properties = calc.calc(samples, label)

        # Calculate cost
        cost = n_structures * self.acquisition_fn.cost_model[self.default_calculator]

        metadata = {
            'routed_to': [self.default_calculator] * n_structures,
            'uncertainties': None,
            'gp_predictions': None,
            'features': None,
            'cost': cost,
            'routing_counts': {self.default_calculator: n_structures},
            'best_observed': None,
        }

        return properties, metadata

    def _get_best_observed(self) -> float:
        """
        Get best observed property value from GP training data.

        Checks multiple sources (in order of preference):
        1. Buffer-based y_train (add_data workflow)
        2. Fitted data from last fit() call
        3. BoTorch model's training targets (fallback)

        Returns:
            float: Maximum observed value (for maximization problems)
        """
        # Path 1: Check buffer (legacy add_data workflow)
        if self.gp_model.y_train:
            all_y = np.concatenate(self.gp_model.y_train) if len(self.gp_model.y_train) > 1 else self.gp_model.y_train[0]
            return float(np.max(all_y))

        # Path 2: Check fitted data from last fit(X, y) call
        if hasattr(self.gp_model, '_y_train_fitted') and self.gp_model._y_train_fitted is not None:
            return float(np.max(self.gp_model._y_train_fitted))

        # Path 3: Extract from BoTorch model if available
        if self.gp_model.model is not None and self.gp_model.is_trained:
            try:
                y_scaled = self.gp_model.model.train_targets.cpu().numpy()
                y_original = self.gp_model.target_scaler.inverse_transform(
                    y_scaled.reshape(-1, 1)
                ).flatten()
                return float(np.max(y_original))
            except Exception as e:
                logging.warning(f'Failed to extract best observed from BoTorch model: {e}')

        # Fallback: return 0.0 (only happens if GP never trained)
        return 0.0

    def _format_routing_counts(self, calc_groups: Dict[str, List[int]], total: int) -> str:
        """Format routing counts for logging."""
        counts_str = ", ".join([
            f"{calc}: {len(indices)} ({100*len(indices)/total:.1f}%)"
            for calc, indices in calc_groups.items()
        ])
        return counts_str
