"""GP Training Manager for online learning during RL.

Manages GP model training lifecycle:
- Collects training data from LTM
- Batch retraining every N steps
- Tracks metrics and performance
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

from rewards.gp.surrogate import GPSurrogate
from rewards.gp.metrics import GPMetrics
from rewards.calculators.orb.featurizer import ORBFeaturizer
from memory.ltm import LongTimeMem


class GPTrainingManager:
    """
    Manages GP model training lifecycle during RL.

    Responsibilities:
    1. Collect training data from LTM (structures + property values)
    2. Retrain GP every N RL steps
    3. Compute and track metrics (RMSE, R², etc.)
    4. Save training history for analysis
    """

    def __init__(
        self,
        gp_model: GPSurrogate,
        featurizer: ORBFeaturizer,
        retrain_frequency: int = 5,
        metrics_dir: Optional[str] = None,
        min_samples: int = 10,
        validation_split: float = 0.2,
        noise_estimator=None  # NoiseEstimator instance for heteroscedastic GP
    ):
        """
        Initialize GP Training Manager.

        Args:
            gp_model: GPSurrogate instance
            featurizer: ORBFeaturizer instance
            retrain_frequency: Retrain every N RL steps
            metrics_dir: Directory to save metrics (optional)
            min_samples: Minimum samples before first training
            validation_split: Fraction of data for validation
            noise_estimator: NoiseEstimator for learning heteroscedastic noise (optional)
        """
        self.gp_model = gp_model
        self.featurizer = featurizer
        self.retrain_frequency = retrain_frequency
        self.metrics_dir = metrics_dir
        self.min_samples = min_samples
        self.validation_split = validation_split
        self.noise_estimator = noise_estimator

        self.training_history = []
        self.last_retrain_step = -1
        self.noise_levels_learned = False

        if self.metrics_dir and not os.path.exists(self.metrics_dir):
            os.makedirs(self.metrics_dir)

    def should_retrain(self, current_step: int) -> bool:
        """
        Check if GP should be retrained at current RL step.

        Args:
            current_step: Current RL epoch

        Returns:
            bool: True if should retrain
        """
        if current_step - self.last_retrain_step >= self.retrain_frequency:
            return True
        return False

    def collect_training_data(
        self,
        ltm: LongTimeMem,
        property_name: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Collect training data from LTM.

        Extracts:
        - Features (ORB embeddings) from LTM or computes them
        - Property values for the specified property

        Args:
            ltm: LongTimeMem instance
            property_name: Name of property column in LTM

        Returns:
            X: Feature matrix (n_samples, feature_dim) or None
            y: Property values (n_samples,) or None
        """
        if len(ltm) < self.min_samples:
            logging.info(f"GP Trainer: Not enough samples ({len(ltm)} < {self.min_samples})")
            return None, None

        # Check if property values exist in LTM
        if property_name not in ltm.memory.columns:
            logging.warning(f"GP Trainer: Property '{property_name}' not in LTM")
            return None, None

        # Get valid samples (non-null property values)
        valid_mask = ltm.memory[property_name].notna()
        valid_structures = ltm.memory.loc[valid_mask, 'struc'].tolist()
        valid_properties = ltm.memory.loc[valid_mask, property_name].values.astype(float)

        if len(valid_properties) < self.min_samples:
            logging.info(f"GP Trainer: Not enough valid samples ({len(valid_properties)} < {self.min_samples})")
            return None, None

        # Get or compute features
        if 'features' in ltm.memory.columns and ltm.memory.loc[valid_mask, 'features'].notna().all():
            # Features cached in LTM
            logging.info("GP Trainer: Using cached features from LTM")
            X = np.stack(ltm.memory.loc[valid_mask, 'features'].values)
        else:
            # Compute features
            logging.info(f"GP Trainer: Computing ORB features for {len(valid_structures)} structures...")
            import pandas as pd
            from pymatgen.io.ase import AseAtomsAdaptor

            adaptor = AseAtomsAdaptor()
            ase_atoms_list = [adaptor.get_atoms(s) for s in valid_structures]
            df = pd.DataFrame({'ase_atoms': ase_atoms_list})

            X = self.featurizer.fit_transform(df) if not self.featurizer.is_fitted else self.featurizer.transform(df)

            # Cache features in LTM for future use
            # (Note: This modifies LTM but is useful for efficiency)
            ltm.memory.loc[valid_mask, 'features'] = list(X)

        y = valid_properties

        logging.info(f"GP Trainer: Collected {len(X)} samples, feature dim: {X.shape[1]}")

        return X, y

    def collect_training_data_with_noise(
        self,
        ltm: LongTimeMem,
        property_name: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Collect training data WITH noise variances from LTM.

        Extends collect_training_data to also extract noise variances
        based on calculator_used and noise_estimator.

        Args:
            ltm: LongTimeMem instance
            property_name: Name of property column in LTM

        Returns:
            X: Feature matrix (n_samples, feature_dim) or None
            y: Property values (n_samples,) or None
            noise_var: Noise variances (n_samples,) or None if noise estimator not available
        """
        # First collect features and targets using existing method
        X, y = self.collect_training_data(ltm, property_name)

        if X is None or y is None:
            return None, None, None

        # If no noise estimator, return None for noise_var
        if self.noise_estimator is None or not self.noise_levels_learned:
            return X, y, None

        # Extract calculator metadata for valid samples
        valid_mask = ltm.memory[property_name].notna()

        if 'calculator_used' not in ltm.memory.columns:
            logging.warning("GP Trainer: 'calculator_used' column missing, cannot use heteroscedastic noise")
            return X, y, None

        calculators = ltm.memory.loc[valid_mask, 'calculator_used'].values

        # Get noise variance array
        noise_var = self.noise_estimator.get_noise_array(calculators)

        logging.info(f"GP Trainer: Collected noise variances, range: [{noise_var.min():.3f}, {noise_var.max():.3f}]")

        return X, y, noise_var

    def retrain(
        self,
        ltm: LongTimeMem,
        current_step: int,
        property_name: str = 'property_value'
    ) -> Dict[str, float]:
        """
        Retrain GP model on data from LTM.

        Args:
            ltm: LongTimeMem instance
            current_step: Current RL epoch
            property_name: Property column name in LTM

        Returns:
            dict: Training and validation metrics
        """
        logging.info(f"\nGP Trainer: Retraining at step {current_step}...")

        # Try to estimate noise levels if not yet learned
        if self.noise_estimator is not None and not self.noise_levels_learned:
            noise_estimates = self.noise_estimator.estimate_from_ltm(ltm)
            if noise_estimates is not None:
                self.noise_levels_learned = True
                logging.info(f"GP Trainer: Noise levels learned from LTM")

                # Save noise estimates
                if self.metrics_dir:
                    noise_save_path = os.path.join(self.metrics_dir, 'noise_estimates.csv')
                    self.noise_estimator.save_estimates(noise_save_path)

        # Collect data (with noise variances if available)
        X, y, noise_var = self.collect_training_data_with_noise(ltm, property_name)

        if X is None or y is None:
            logging.warning(f"GP Trainer: Insufficient data for training at step {current_step}")
            return {}

        # Train/val split
        n_samples = len(X)
        n_val = int(n_samples * self.validation_split)

        np.random.seed(current_step)  # Reproducible splits
        indices = np.random.permutation(n_samples)
        train_idx, val_idx = indices[n_val:], indices[:n_val]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        if noise_var is not None:
            noise_var_train = noise_var[train_idx]
        else:
            noise_var_train = None

        logging.info(f"GP Trainer: Train: {len(X_train)}, Val: {len(X_val)}")

        # Train GP (with or without noise variances)
        self.gp_model.fit(X_train, y_train, noise_var=noise_var_train)

        # Evaluate
        train_metrics = self._evaluate(X_train, y_train, prefix='train')
        val_metrics = self._evaluate(X_val, y_val, prefix='val') if len(X_val) > 0 else {}

        # Combined metrics
        metrics = {**train_metrics, **val_metrics}
        metrics['step'] = current_step
        metrics['n_train_samples'] = len(X_train)
        metrics['n_val_samples'] = len(X_val)

        # Log metrics
        self.training_history.append(metrics)
        self.last_retrain_step = current_step

        logging.info(f"GP Trainer: Retraining complete!")
        if val_metrics:
            logging.info(f"  Val R²: {val_metrics['val_R2']:.4f}, RMSE: {val_metrics['val_RMSE']:.4f}")

        # Save metrics
        if self.metrics_dir:
            self._save_metrics()

        return metrics

    def _evaluate(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        prefix: str = ''
    ) -> Dict[str, float]:
        """
        Evaluate GP model and compute metrics.

        Args:
            X: Feature matrix
            y_true: True property values
            prefix: Prefix for metric keys (e.g., 'train' or 'val')

        Returns:
            dict: Metrics with prefix
        """
        y_pred, y_std = self.gp_model.predict(X, return_std=True)
        metrics = GPMetrics.compute_metrics(y_true, y_pred.flatten(), y_std.flatten())

        # Add prefix to keys
        if prefix:
            metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}

        return metrics

    def _save_metrics(self):
        """Save training history to CSV."""
        if not self.training_history:
            return

        df = pd.DataFrame(self.training_history)
        save_path = os.path.join(self.metrics_dir, 'gp_training_metrics.csv')
        df.to_csv(save_path, index=False)
        logging.info(f"GP Trainer: Saved metrics to {save_path}")
