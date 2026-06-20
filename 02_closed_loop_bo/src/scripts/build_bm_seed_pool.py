"""Build ltm_bm_seed_pool.parquet from matminer's elastic_tensor_2015.

Output: $SCRATCH/matinvent-hcap-bo/data/bm/ltm_bm_seed_pool.parquet

Workflow:
    1. matminer.load_dataset('elastic_tensor_2015') — 1,181 MP structures with K_VRH
    2. drop NaN K_VRH
    3. spglib.standardize_cell → ASE Atoms (matches FME paper preprocessing)
    4. random sample 500 (seed=42)
    5. ORB-PCA50 featurize (same featurizer the live GP uses)
    6. write LTM rows: {structure_id, formula, cycle_id=-1, atoms_json,
                        Z_pca50, y_cp (=K_VRH GPa), y_cp_var=NaN,
                        sigma_pred=NaN, ood_score=NaN, oracle_source='bm_seed'}

Columns named `y_cp` for compatibility with the LTM schema; the value is
K_VRH in GPa for BM runs (no schema change needed).
"""
import os
import sys
import numpy as np
import pandas as pd

OUT_DIR = os.path.expandvars("$SCRATCH/matinvent-hcap-bo/data/bm")
OUT = os.path.join(OUT_DIR, "ltm_bm_seed_pool.parquet")
N_SAMPLES = 500
SEED = 42


def _standardize(atoms):
    import spglib
    from ase import Atoms
    cell = (atoms.get_cell(), atoms.get_scaled_positions(), atoms.get_atomic_numbers())
    res = spglib.standardize_cell(cell, to_primitive=True, symprec=1e-3)
    if res is None:
        return atoms
    new_cell, new_pos, new_numbers = res
    return Atoms(numbers=new_numbers, scaled_positions=new_pos, cell=new_cell, pbc=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading matminer elastic_tensor_2015 ...", flush=True)
    from matminer.datasets import load_dataset
    df = load_dataset("elastic_tensor_2015")
    print(f"  raw rows: {len(df)}  columns include K_VRH={'K_VRH' in df.columns}", flush=True)

    # Drop missing K_VRH and structure
    df = df.dropna(subset=["K_VRH", "structure"]).reset_index(drop=True)
    print(f"  after dropna: {len(df)}", flush=True)
    print(f"  K_VRH range: {df['K_VRH'].min():.1f} .. {df['K_VRH'].max():.1f} GPa  "
          f"median={df['K_VRH'].median():.1f}", flush=True)

    # pymatgen Structure → ASE Atoms (standardized to primitive)
    from pymatgen.io.ase import AseAtomsAdaptor
    adaptor = AseAtomsAdaptor()
    atoms_list = []
    K_list = []
    formulas = []
    n_failed = 0
    for _, row in df.iterrows():
        try:
            a = adaptor.get_atoms(row["structure"])
            a = _standardize(a)
            atoms_list.append(a)
            K_list.append(float(row["K_VRH"]))
            formulas.append(a.get_chemical_formula())
        except Exception as e:
            n_failed += 1
    print(f"  standardized: {len(atoms_list)}  failed: {n_failed}", flush=True)

    # Random sample 500
    rng = np.random.default_rng(SEED)
    if len(atoms_list) > N_SAMPLES:
        idx = rng.permutation(len(atoms_list))[:N_SAMPLES]
        atoms_list = [atoms_list[i] for i in idx]
        K_list = [K_list[i] for i in idx]
        formulas = [formulas[i] for i in idx]
    print(f"  sampled: {len(atoms_list)}", flush=True)

    # ORB-PCA50 featurize
    print("Featurizing with ORB-PCA50 ...", flush=True)
    sys.path.insert(0, os.path.expandvars("$SCRATCH/matinvent-hcap-bo"))
    from src.featurizer import ORBFeaturizer
    feat = ORBFeaturizer(n_components=50, device=os.environ.get("GP_DEVICE", "cuda"))
    Z = feat.fit_transform(atoms_list)
    print(f"  Z shape: {Z.shape}", flush=True)

    # Build LTM rows
    from src.ltm import canonical_atoms_id, atoms_to_json
    rows = []
    for atoms, K, formula, z in zip(atoms_list, K_list, formulas, Z):
        rows.append({
            "structure_id": canonical_atoms_id(atoms),
            "formula": formula,
            "cycle_id": -1,
            "atoms_json": atoms_to_json(atoms),
            "Z_pca50": list(map(float, z)),
            "y_cp": float(K),  # K_VRH in GPa (column reused)
            "y_cp_var": float("nan"),
            "sigma_pred": float("nan"),
            "ood_score": float("nan"),
            "oracle_source": "bm_seed_elastic_tensor_2015",
        })

    df_out = pd.DataFrame(rows)
    df_out.to_parquet(OUT, index=False)
    print(f"\nWrote {OUT}", flush=True)
    print(f"  rows: {len(df_out)}", flush=True)
    print(f"  K_VRH min={df_out['y_cp'].min():.1f}  "
          f"median={df_out['y_cp'].median():.1f}  "
          f"max={df_out['y_cp'].max():.1f}  GPa", flush=True)
    print("\nTop 10 by K_VRH:", flush=True)
    print(df_out[["formula", "y_cp"]].sort_values("y_cp", ascending=False)
          .head(10).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
