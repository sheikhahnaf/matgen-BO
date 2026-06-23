# DFT validation — AUTO-SYNCED results

_Regenerated each sync from FASTER result JSONs. Tables only — do not hand-edit; curated narrative lives in `insights.md`._

## K0 — bulk modulus (26/29 done)

| structure | K0 (GPa) | V0 (A^3) | ISYM | |M| | in-win | smearing |
|---|---|---|---|---|---|---|
| bm_top03_mg_ACC_seed113_Os5W3_sg65 | 355.9 | 120.67 | 2 | 0.00 | True | tetra-5 |
| bm_top01_adit_ACC_seed7_MoN_sg187 | 353.6 | 80.85 | 2 | 0.00 | True | tetra-5 |
| bm_top08_cf_ACC_seed99_Re5W_sg1 | 351.7 | 183.26 | 0 | 0.00 | True | tetra-5 |
| bm_top07_mg_ACC_seed99_CoIrOs2_sg119 | 348.1 | 54.09 | 2 | 0.31 | True | tetra-5 |
| bm_top02_adit_BASE_seed23_MoC_sg194 | 348.1 | 83.70 | 2 | 0.00 | True | tetra-5 |
| bm_top05_mg_ACC_seed99_CoIrOs2_sg123 | 346.0 | 54.07 | 2 | 0.00 | True | tetra-5 |
| bm_top06_mg_ACC_seed99_CoIrOs2_sg123 | 346.0 | 54.07 | 2 | 0.00 | True | tetra-5 |
| bm_top09_cf_BASE_seed99_VIr7_sg187 | 334.6 | 114.71 | 2 | 0.00 | True | tetra-5 |
| bm_top11_adit_ACC_seed17_TaB2W_sg38 | 315.2 | 43.65 | 2 | 0.00 | True | tetra-5 |
| bm_top13_adit_BASE_seed17_FeB2MoW_sg26 | 306.9 | 105.15 | 2 | 3.71 | True | tetra-5 |
| bm_top14_adit_BASE_seed17_FeB2MoW_sg26 | 306.9 | 105.15 | 2 | 3.71 | True | tetra-5 |
| bm_top16_adit_BASE_seed23_Mn(BMo)3_sg63 | 300.5 | 146.32 | 2 | 2.38 | True | tetra-5 |
| bm_top19_cf_ACC_seed113_ReIr2Rh5_sg5 | 299.5 | 113.81 | 2 | 0.00 | True | tetra-5 |
| bm_top17_adit_ACC_seed23_FeRe(PW)2_sg26 | 293.4 | 159.04 | 2 | 0.00 | True | tetra-5 |
| cp_top19_adit_BASE_seed17_Li3PO4_sg1 | 131.3 | 163.37 | 0 | 0.00 | True | tetra-5 |
| cp_top16_cf_BASE_seed113_Li6TiO5_sg1 | 101.0 | 124.35 | 0 | 0.00 | True | tetra-5 |
| cp_top17_adit_ACC_seed23_LiMg2_sg40 | 29.5 | 128.50 | 2 | 0.00 | True | tetra-5 |
| cp_top14_cf_ACC_seed7_LiMg_sg6 | 25.3 | 125.65 | 2 | 0.00 | True | tetra-5 |
| cp_top05_mg_BASE_seed23_Li13Mg3Al2_sg1 | 23.7 | 341.97 | 0 | 0.00 | True | tetra-5 |
| cp_top07_mg_BASE_seed113_Li7Mg3_sg1 | 20.6 | 408.82 | 0 | 0.00 | True | tetra-5 |
| cp_top13_cf_ACC_seed99_Li5CaMg2_sg6 | 19.0 | 186.43 | 2 | 0.00 | True | tetra-5 |
| cp_top03_mg_ACC_seed23_Li4Mg_sg139 | 18.5 | 100.94 | 2 | 0.00 | True | tetra-5 |
| cp_top02_mg_ACC_seed7_Li4Mg_sg12 | 18.2 | 202.31 | 2 | 0.00 | True | tetra-5 |
| cp_top01_mg_BASE_seed23_Li13Mg3_sg38 | 18.2 | 324.26 | 2 | 0.00 | True | tetra-5 |
| cp_top04_cf_BASE_seed7_NaLi5_sg1 | 11.8 | 142.47 | 0 | 0.00 | True | tetra-5 |
| cp_top06_cf_ACC_seed23_NaLi2_sg11 | 11.2 | 154.05 | 2 | 0.00 | True | tetra-5 |

## Phonons — Cv(300K) / dynamical stability (2 done)

| structure | supercell | Cv300 (J/g/K) | min_freq (THz) | n_imag | stable |
|---|---|---|---|---|---|
| Si_prim_POSCAR_L10 | [3, 3, 3] | 0.7132 | 0.306 | 0 | True |
| Si_prim_POSCAR_L14 | [4, 4, 4] | 0.7139 | 0.295 | 0 | True |

## Convergence study

### k-convergence: bm_top01_adit_ACC_seed7_MoN_sg187 (ISYM=2)
| KSPACING | B0 (GPa) | grid | smearing |
|---|---|---|---|
| 0.30 | 353.2 | [9, 9, 2] | gauss-0/0.05 |
| 0.24 | 353.7 | [11, 11, 3] | gauss-0/0.05 |
| 0.20 | 353.7 | [13, 13, 3] | gauss-0/0.05 |
| 0.16 | 353.7 | [16, 16, 4] | tetra-5 |
| 0.12 | 353.6 | [22, 22, 5] | tetra-5 |
### k-convergence: cp_top02_mg_ACC_seed7_Li4Mg_sg12 (ISYM=2)
| KSPACING | B0 (GPa) | grid | smearing |
|---|---|---|---|
| 0.30 | 18.5 | [5, 5, 4] | tetra-5 |
| 0.24 | 18.2 | [6, 6, 5] | tetra-5 |
| 0.20 | 18.4 | [7, 7, 6] | tetra-5 |
| 0.16 | 18.2 | [9, 9, 7] | tetra-5 |
| 0.12 | 18.3 | [12, 12, 9] | tetra-5 |
### ENCUT+KSPACING conv: cp_top19_adit_BASE_seed17_Li3PO4_sg1 (ISYM=0)
ENCUT sweep (KSPACING=0.16): 520->130.4
### ENCUT+KSPACING conv: cp_top20_adit_ACC_seed17_Na2BO3_sg1 (ISYM=0)
ENCUT sweep (KSPACING=0.16): 
KSPACING sweep (ENCUT=680): 
