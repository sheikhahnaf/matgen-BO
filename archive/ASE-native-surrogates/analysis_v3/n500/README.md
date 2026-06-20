# Simplified ASE Regression Analysis (n=500)

## Overview

This directory contains a **simplified, focused analysis** of ASE (Atomic Simulation Environment) regression test results for predicting materials properties using surrogate models.

**Key Design Principles:**
1. **Focus on n=500 ONLY** - Since DGP only has complete data for this training set size
2. **Automatic Best PCA Selection** - No averaging or fixed PCA values; automatically selects optimal PCA per surrogate
3. **Simplified Visualizations** - Essential plots only (bar charts, heatmaps, radar charts, PCA sensitivity)
4. **Extensible** - Easy to add n=100, 250 when DGP data becomes available

---

## Quick Start

### Run All Analyses

```bash
cd analysis_v2/scripts
python run_all.py
```

This will execute all 7 phases in sequence (~40 seconds total).

### Run Individual Phases

```bash
cd analysis_v2/scripts

# Phase 0: Data Preparation
python prepare_data.py

# Phase 1: Bar Charts (Averaged)
python bar_charts_averaged.py

# Phase 2: Bar Charts (Per Property)
python bar_charts_per_property.py

# Phase 3: Heatmaps
python heatmaps_averaged.py

# Phase 4: Property Difficulty Matrix
python property_difficulty.py

# Phase 5: PCA Sensitivity Analysis
python pca_sensitivity.py

# Phase 6: Radar Charts (ORB Only)
python radar_charts.py
```

---

## Analysis Phases

### Phase 0: Data Preparation

**Script:** `prepare_data.py`

**Purpose:** Load and filter aggregated results to n=500 only

**Filters Applied:**
- `n_train = 500` (excluding 100, 250)
- Remove `kpoint_density` property (excluded from analysis)
- Keep only `R²`, `RMSE`, `Spearman` metrics (exclude MAE, SMAPE)

**Output:** `data/filtered_n500.csv` (864 rows)

**Data Structure:**
- 3 models: GP, MTGP_2, DGP
- 4 descriptors: MACE, ORB, SOAP, UMA
- 3 PCA values: 10, 25, 50
- 8 properties: K_Voigt, K_VRH, K_Reuss, G_Voigt, G_VRH, G_Reuss, elastic_anisotropy, poisson_ratio
- 3 metrics: R², RMSE, Spearman

Total: 3 × 4 × 3 × 8 × 3 = 864 rows ✓

---

### Phase 1: Bar Charts (Averaged Across Properties)

**Script:** `bar_charts_averaged.py`

**Purpose:** Compare surrogates using **best PCA per model-descriptor combo** averaged across all 8 properties

**Algorithm:**
1. For each (model, descriptor) combination:
   - Compute average R² across 8 properties for PCA=10, 25, 50
   - Select PCA with highest average R²
2. Use this "best average PCA" for all metrics (R², RMSE, Spearman)
3. Create bar charts comparing models

**Outputs:**
- `figures/bar_charts/averaged_R2_n500.pdf`
- `figures/bar_charts/averaged_RMSE_n500.pdf`
- `figures/bar_charts/averaged_Spearman_n500.pdf`
- `data/best_pca_averaged.csv` (table of PCA choices)

**Key Finding:**
- GP and MTGP prefer PCA=50 for all descriptors
- DGP shows variation: PCA=10 for MACE, PCA=25 for others

---

### Phase 2: Bar Charts (Per Property)

**Script:** `bar_charts_per_property.py`

**Purpose:** Same as Phase 1, but separate analysis for each property

**Key Difference:** Best PCA is selected **per property**, not averaged. This means different properties may prefer different PCA values even for the same model-descriptor combo.

**Outputs:**
- `figures/bar_charts/per_property/{property}_{metric}_n500.pdf` (24 files)
- `data/best_pca_per_property.csv` (table of PCA choices per property)

---

### Phase 3: Heatmaps (Averaged with Spearman)

**Script:** `heatmaps_averaged.py`

**Purpose:** Heatmap visualization of averaged performance using best PCA from Phase 1

**Heatmap Structure:**
- Rows: Descriptors (ORB, MACE, UMA, SOAP)
- Columns: Models (GP, MTGP, DGP)
- Color: Metric value (R², RMSE, or Spearman)

**Outputs:**
- `figures/heatmaps/averaged_R2_n500.pdf`
- `figures/heatmaps/averaged_RMSE_n500.pdf`
- `figures/heatmaps/averaged_Spearman_n500.pdf`

**Key Finding:**
- ORB descriptor consistently best across all models
- SOAP descriptor worst across all models

---

### Phase 4: Property Difficulty Matrix

**Script:** `property_difficulty.py`

**Purpose:** Identify which properties are easiest/hardest to predict **for each surrogate separately**

**Analysis:** 3 heatmaps (one per model) showing R² for each (descriptor, property) pair

**Outputs:**
- `figures/property_difficulty/difficulty_matrix_per_surrogate_n500.pdf`
- `data/property_difficulty_per_surrogate.csv`

**Key Findings:**

**Easiest Properties (all models):**
1. Bulk moduli: K_Voigt, K_VRH, K_Reuss (R² ≈ 0.68-0.83)
2. Shear moduli: G_Voigt, G_VRH, G_Reuss (R² ≈ 0.54-0.70)

**Hardest Properties:**
- GP: elastic_anisotropy (R² = 0.47), poisson_ratio (R² = 0.49)
- MTGP: poisson_ratio (R² = 0.37), elastic_anisotropy (R² = 0.50)
- DGP: **poisson_ratio (R² = 0.16)**, **elastic_anisotropy (R² = 0.26)** ← DGP struggles significantly!

---

### Phase 5: PCA Sensitivity Analysis

**Script:** `pca_sensitivity.py`

**Purpose:** Understand how PCA choice affects performance

**Two Analyses:**

#### 5.1 Averaged Across Properties
- Shows how average R²/Spearman changes with PCA (10 → 25 → 50)
- 4 subplots (one per descriptor)

#### 5.2 Per Property (ORB only)
- Shows PCA sensitivity for each of 8 properties
- Uses ORB descriptor only (best performer)

**Outputs:**
- `figures/pca_sensitivity/averaged_R2_n500.pdf`
- `figures/pca_sensitivity/averaged_Spearman_n500.pdf`
- `figures/pca_sensitivity/per_property_R2_n500.pdf`
- `figures/pca_sensitivity/per_property_Spearman_n500.pdf`
- `data/pca_sensitivity_averaged.csv`
- `data/pca_sensitivity_per_property.csv`

**Key Findings:**

**GP:** Low PCA sensitivity
- ORB: R² range = 0.07 (stable)
- MACE: R² range = 0.01 (very stable)
- UMA, SOAP: Higher sensitivity (ranges ≈ 0.25)

**MTGP:** Moderate PCA sensitivity
- ORB: R² range = 0.12
- UMA: R² range = 0.46 (high sensitivity)

**DGP:** **EXTREME PCA sensitivity** 🚨
- ORB: R² range = 0.82 (goes from 0.58 at PCA10, peaks at 0.81 at PCA25, crashes to -0.02 at PCA50!)
- MACE: R² range = 0.58 (similar crash at PCA50)
- **ALL properties** show high sensitivity (ranges 0.6-0.95)

**Interpretation:** DGP is unstable at high PCA values. This is why Phase 1 selected lower PCA for DGP.

---

### Phase 6: Radar Charts (ORB Featurizer Only)

**Script:** `radar_charts.py`

**Purpose:** Compare surrogates across properties using radar plots

**Design:**
- **Vertices:** 8 properties (not models!)
- **Lines:** 3 surrogates (GP, MTGP, DGP)
- **Descriptor:** ORB only (best performer)
- **PCA:** Best per surrogate per property

**Outputs:**
- `figures/radar_charts/orb_R2_n500.pdf`
- `figures/radar_charts/orb_Spearman_n500.pdf`
- `data/radar_orb_pca_choices.csv`

**Key Findings:**

**Average Performance (ORB, R²):**
- DGP: 0.807 (highest)
- GP: 0.803
- MTGP: 0.780

**Average Performance (ORB, Spearman):**
- GP: 0.844 (highest)
- MTGP: 0.843
- DGP: 0.826

**Best/Worst Properties:**
- All models: Best = K_Voigt (R² ≈ 0.89-0.93)
- GP: Worst = elastic_anisotropy (R² = 0.58)
- MTGP/DGP: Worst = poisson_ratio (R² ≈ 0.59-0.62)

**PCA Usage (ORB):**
- GP: 100% use PCA=50
- MTGP: 100% use PCA=50
- DGP: 100% use PCA=25 ← Avoids instability at PCA=50

---

## Directory Structure

```
analysis_v2/
├── README.md                          # This file
│
├── scripts/                           # Analysis scripts
│   ├── prepare_data.py                # Phase 0
│   ├── bar_charts_averaged.py         # Phase 1
│   ├── bar_charts_per_property.py     # Phase 2
│   ├── heatmaps_averaged.py           # Phase 3
│   ├── property_difficulty.py         # Phase 4
│   ├── pca_sensitivity.py             # Phase 5
│   ├── radar_charts.py                # Phase 6
│   └── run_all.py                     # Master script
│
├── data/                              # Generated data tables
│   ├── filtered_n500.csv                      # Filtered input data (864 rows)
│   ├── best_pca_averaged.csv                  # Best PCA per surrogate (averaged)
│   ├── best_pca_per_property.csv              # Best PCA per property
│   ├── property_difficulty_per_surrogate.csv  # Difficulty analysis
│   ├── pca_sensitivity_averaged.csv           # PCA sensitivity (averaged)
│   ├── pca_sensitivity_per_property.csv       # PCA sensitivity (per property)
│   └── radar_orb_pca_choices.csv              # PCA choices for radar charts
│
└── figures/                           # Generated visualizations
    ├── bar_charts/
    │   ├── averaged_R2_n500.pdf               # Phase 1 outputs (3 files)
    │   ├── averaged_RMSE_n500.pdf
    │   ├── averaged_Spearman_n500.pdf
    │   └── per_property/                      # Phase 2 outputs (24 files)
    │       ├── K_Voigt_R2_n500.pdf
    │       ├── K_Voigt_RMSE_n500.pdf
    │       └── ... (21 more)
    │
    ├── heatmaps/                              # Phase 3 outputs (3 files)
    │   ├── averaged_R2_n500.pdf
    │   ├── averaged_RMSE_n500.pdf
    │   └── averaged_Spearman_n500.pdf
    │
    ├── property_difficulty/                   # Phase 4 output (1 file)
    │   └── difficulty_matrix_per_surrogate_n500.pdf
    │
    ├── pca_sensitivity/                       # Phase 5 outputs (4 files)
    │   ├── averaged_R2_n500.pdf
    │   ├── averaged_Spearman_n500.pdf
    │   ├── per_property_R2_n500.pdf
    │   └── per_property_Spearman_n500.pdf
    │
    └── radar_charts/                          # Phase 6 outputs (2 files)
        ├── orb_R2_n500.pdf
        └── orb_Spearman_n500.pdf
```

**Total Outputs:**
- Data: 7 CSV files
- Figures: 37 PDF files (3 + 24 + 3 + 1 + 4 + 2)

---

## Key Scientific Insights

### 1. Descriptor Performance

**Ranking (by averaged R²):**
1. **ORB** (R² ≈ 0.78-0.81) - Consistently best across all models
2. **MACE/UMA** (R² ≈ 0.54-0.73) - Moderate performance
3. **SOAP** (R² ≈ 0.25-0.45) - Worst performance

**Recommendation:** Use ORB descriptor for materials property prediction.

---

### 2. Model Performance

**GP (Gaussian Process):**
- **Strengths:** Stable, consistent, low PCA sensitivity
- **R² (ORB):** 0.803 averaged, 0.844 Spearman
- **PCA preference:** 50 (always)
- **Best for:** Reliable predictions with good uncertainty quantification

**MTGP (Multi-Task GP):**
- **Strengths:** Reasonable multi-task learning
- **R² (ORB):** 0.780 averaged, 0.843 Spearman
- **PCA preference:** 50 (always)
- **Weakness:** Lower R² than GP for most properties

**DGP (Deep Gaussian Process):**
- **Strengths:** Highest R² on some properties
- **R² (ORB):** 0.807 averaged, 0.826 Spearman
- **Critical Issue:** **EXTREME PCA sensitivity!**
  - Performance crashes at PCA=50 (R² goes negative!)
  - Requires careful PCA tuning (prefers PCA=25)
  - Extremely unstable for poisson_ratio (R² = 0.16) and elastic_anisotropy (R² = 0.26)
- **Recommendation:** Use with caution; extensive hyperparameter tuning required

---

### 3. Property Difficulty

**Easy Properties (R² > 0.7 for all models):**
- Bulk moduli: K_Voigt, K_VRH, K_Reuss
- Rationale: These are fundamental, well-defined elastic properties with low intrinsic noise

**Moderate Properties (0.5 < R² < 0.7):**
- Shear moduli: G_Voigt, G_VRH, G_Reuss
- Rationale: More sensitive to local atomic environments

**Hard Properties (R² < 0.5 for some models):**
- elastic_anisotropy (GP: 0.47, DGP: 0.26)
- poisson_ratio (MTGP: 0.37, DGP: 0.16)
- Rationale: Derived properties that depend on ratios/combinations of elastic constants; higher noise amplification

---

### 4. PCA Selection Strategy

**GP/MTGP:** Use PCA=50 (maximum)
- Low sensitivity to PCA choice
- Higher PCA generally better

**DGP:** Use PCA=25 (medium)
- **DO NOT use PCA=50** - model crashes!
- Requires property-specific tuning
- Consider separate PCA optimization per property

---

## Future Extensions

When DGP data for n=100 and n=250 becomes available, extend this analysis:

### New Analyses to Add:

1. **Multi-size comparison**
   - Run same analyses for n=100, 250, 500
   - Compare how "best PCA" changes with dataset size

2. **Learning curves**
   - Performance vs dataset size (now possible!)
   - Understand sample efficiency per model

3. **Size-dependent PCA optimization**
   - Does optimal PCA decrease with smaller datasets?
   - Property-specific sample size requirements

### Code Modifications Needed:

Minimal! Just add `dataset_size` parameter:

```python
# Current
def get_best_pca_per_surrogate_averaged(df, metric='R2'):
    ...

# Future
def get_best_pca_per_surrogate_averaged(df, dataset_size=500, metric='R2'):
    df_filtered = df[df['n_train'] == dataset_size]
    ...
```

Then loop over `[100, 250, 500]` instead of just `500`.

---

## Dependencies

**Python packages:**
- pandas (data manipulation)
- numpy (numerical operations)
- matplotlib (plotting)
- seaborn (statistical visualizations)

**Data source:**
- Input: `../ASE_regression_test/analysis/aggregated_results.csv`
- Generated by previous aggregation pipeline

---

## Reproducibility

All analyses are **fully reproducible**:

1. Fixed random seeds (not applicable - no randomness)
2. Deterministic PCA selection (maximum or minimum over fixed set)
3. Fixed data filtering criteria
4. Version-controlled scripts

**To reproduce:**
```bash
cd analysis_v2/scripts
python run_all.py
```

Output will be identical (assuming same input data).

---

## Differences from Previous Analysis

| Aspect | Previous (analysis/) | This (analysis_v2/) |
|--------|---------------------|---------------------|
| Dataset sizes | n=100, 250, 500 | **n=500 ONLY** |
| DGP data | Incomplete (only n=500) | Complete (all models have n=500) |
| PCA selection | Average across all PCAs | **Best PCA automatically selected** |
| Metrics | R², RMSE, MAE, SMAPE, Spearman | **R², RMSE, Spearman** |
| Properties | 9 (including kpoint_density) | **8 (excluding kpoint_density)** |
| Visualizations | Many complex plots | **Simplified, focused plots** |
| Extensibility | Hard to extend | **Easy to extend when n=100,250 DGP data arrives** |

---

## Citation

If you use this analysis in your research, please cite:

```bibtex
@software{ase_regression_analysis_v2,
  title={Simplified ASE Regression Analysis for Materials Property Prediction},
  author={[Your Name]},
  year={2024},
  url={[Repository URL]}
}
```

---

## Contact

For questions or issues, please contact [your email] or open an issue on [repository URL].

---

**Last Updated:** 2024-02-24
**Analysis Version:** 2.0
**Data Version:** n=500 complete dataset
