"""Re-run the static-benchmark single-task GP fits and keep every holdout
prediction together with its predictive standard deviation.

The published benchmark stored summary metrics only (R2, RMSE, MAE, SMAPE,
Spearman per cell), so posterior calibration could not be computed from the
saved files. This driver calls the unmodified benchmark module
(`gp_regression.main`) with one function replaced: the per-property split loop
is reimplemented without the parity plots and records, for every holdout
sample, the true value, the posterior mean, the latent posterior standard
deviation (the quantity the original code used for its error bars) and the
predictive standard deviation including the fitted observation noise.

Splits, scalers, model and fitting calls are the ones in gp_regression.py, so
the holdout R2 written to gp_<descriptor>_holdout_summary.csv can be compared
directly with the published summary files.

Usage:
    python gp_calibration_rerun.py <src_dir> [gp_regression arguments...]

    <src_dir> holds gp_regression.py and common.py:
        01_static_benchmark/src                 elastic, dielectric (matminer)
        01_static_benchmark/src/phonon_thermo   DFPT phonon thermodynamics (pickle)
"""

import importlib
import os
import sys

import numpy as np
import pandas as pd
import torch


def main():
    src_dir = os.path.abspath(sys.argv[1])
    passthrough = sys.argv[2:]
    sys.path.insert(0, src_dir)
    G = importlib.import_module('gp_regression')
    args = G.make_parser('gp').parse_args(passthrough)

    records = []

    def train_property_multi_split(X_train_all, y_train_all, X_holdout, y_holdout,
                                   device, property_name, n_splits=5, train_ratio=0.8,
                                   output_dir="plots"):
        # output_dir is <out_root>/plots/<descriptor>/<property>
        descriptor = os.path.basename(os.path.dirname(output_dir))
        all_test_metrics, all_holdout_metrics = [], []
        n_total = len(y_train_all)
        n_train = int(train_ratio * n_total)
        print(f"\n  Training samples: {n_total}, Holdout samples: {len(y_holdout)}")

        for split_idx in range(n_splits):
            np.random.seed(split_idx * 42)
            perm = np.random.permutation(n_total)
            train_idx, test_idx = perm[:n_train], perm[n_train:]

            feature_scaler = G.StandardScaler()
            X_train_scaled = feature_scaler.fit_transform(X_train_all[train_idx])
            X_test_scaled = feature_scaler.transform(X_train_all[test_idx])

            target_scaler = G.StandardScaler()
            y_train_scaled = target_scaler.fit_transform(
                y_train_all[train_idx].reshape(-1, 1)).flatten()

            X_train_t = torch.tensor(X_train_scaled, device=device, dtype=torch.float64)
            X_test_t = torch.tensor(X_test_scaled, device=device, dtype=torch.float64)
            y_train_t = torch.tensor(y_train_scaled, device=device)

            model = G.train_gp(X_train_t, y_train_t, device, verbose=False)
            model.eval()
            scale = float(target_scaler.scale_[0])

            with torch.no_grad():
                test_post = model.posterior(X_test_t)
                test_pred = target_scaler.inverse_transform(
                    test_post.mean.squeeze(-1).detach().cpu().numpy().reshape(-1, 1)).flatten()
            all_test_metrics.append(G.calculate_metrics(y_train_all[test_idx], test_pred))

            if X_holdout is not None and len(X_holdout) > 0:
                X_holdout_t = torch.tensor(
                    feature_scaler.transform(X_holdout), device=device, dtype=torch.float64)
                with torch.no_grad():
                    post = model.posterior(X_holdout_t)
                    post_obs = model.posterior(X_holdout_t, observation_noise=True)
                    mu = target_scaler.inverse_transform(
                        post.mean.squeeze(-1).detach().cpu().numpy().reshape(-1, 1)).flatten()
                    sd_latent = post.variance.squeeze(-1).sqrt().detach().cpu().numpy() * scale
                    sd_obs = post_obs.variance.squeeze(-1).sqrt().detach().cpu().numpy() * scale

                holdout_metrics = G.calculate_metrics(y_holdout, mu)
                all_holdout_metrics.append(holdout_metrics)
                print(f"    split {split_idx + 1}: HOLDOUT R2 {holdout_metrics['R2']:.4f}")

                records.append(pd.DataFrame({
                    'descriptor': descriptor,
                    'property': property_name,
                    'split': split_idx,
                    'y_true': np.asarray(y_holdout, dtype=float),
                    'mu': mu,
                    'sd_latent': sd_latent,
                    'sd_obs': sd_obs,
                    'y_train_std': scale,
                }))

        return {
            'test': G.aggregate_metrics(all_test_metrics),
            'holdout': G.aggregate_metrics(all_holdout_metrics) if all_holdout_metrics else None,
            'all_holdout_metrics': all_holdout_metrics,
        }

    G.train_property_multi_split = train_property_multi_split
    sys.argv = ['gp_regression.py'] + passthrough
    G.main()

    out_root = args.output_dir or os.path.join('results', 'gp', args.dataset)
    if records:
        df = pd.concat(records, ignore_index=True)
        dataset = os.path.splitext(os.path.basename(str(args.dataset)))[0]
        df.insert(0, 'n_train', args.n_train)
        df.insert(0, 'pca', args.pca_components)
        df.insert(0, 'dataset', dataset)
        path = os.path.join(out_root, f'calib_preds_{args.descriptor}.csv.gz')
        df.to_csv(path, index=False, float_format='%.8g')
        print(f"\nSaved {len(df)} holdout predictions to: {path}")


if __name__ == '__main__':
    main()
