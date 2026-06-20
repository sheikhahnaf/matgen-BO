#!/usr/bin/env python3
"""Generate ACES SLURM scripts for the phonon/thermo regression benchmark.

Emits the .slurm files into the directory this script lives in. No external deps.

Structure mirrors the original ASE_regression_test/run_*.slurm templates:
  - .env / HF_TOKEN sourcing block
  - env-var routing of descriptors to two python interpreters (MACE vs FairChem)
  - inner PCA x n_train loops (or PCA-only loop for the per-n DGP scripts)
  - output dirs named <dataset>_pca${PCA}_n${NTRAIN}; mtgp uses results/mtgp_2

ACES-specific facts (override the original Grace headers):
  - partition gpu, gres gpu:h100:1, account 156192594849
  - module load CUDA/12.4.0 GCC/13.2.0 WebProxy/0000 (WebProxy kept: needed on ACES nodes)
  - FEAT_CACHE export
  - python envs at /scratch/user/u.sa119259/envs/{ase-test-mace,ase-test}
  - HOLDOUT_CAP=3000 exported ONLY in Arm B (pheasy) scripts; absent in Arm A (dfpt)
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))

# --- Fixed ACES facts -------------------------------------------------------
ACCOUNT = "156192594849"
ROOT = "/scratch/user/u.sa119259/phonon_thermo_benchmark"
FEAT_CACHE = "/scratch/user/u.sa119259/phonon_thermo_benchmark/data/feat_cache"
PY_MACE = "/scratch/user/u.sa119259/envs/ase-test-mace/bin/python"   # soap, mace
PY_FAIR = "/scratch/user/u.sa119259/envs/ase-test/bin/python"        # orb, uma

# --- Datasets ---------------------------------------------------------------
DFPT_PKL = f"{ROOT}/data/dfpt_phonon_thermo.pkl"
PHEASY_PKL = f"{ROOT}/data/pheasy_phonon_thermo.pkl"

# --- Walltimes --------------------------------------------------------------
WALL_GP = "04:00:00"
WALL_MTGP = "12:00:00"
WALL_DGP = "24:00:00"

# --- Model -> output root dir name -----------------------------------------
OUT_ROOT = {"gp": "gp", "mtgp": "mtgp_2", "dgp": "dgp"}

# Map descriptor -> python-env variable name used inside the script.
PY_FOR_DESC = {
    "mace": "PY_MACE",
    "soap": "PY_MACE",
    "orb": "PY_FAIR",
    "uma": "PY_FAIR",
}


def header(job_name, walltime):
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time={walltime}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account={ACCOUNT}
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
"""


def preamble(holdout_cap):
    """Modules, .env/HF_TOKEN sourcing, FEAT_CACHE, optional HOLDOUT_CAP, env paths."""
    lines = []
    lines.append("module purge")
    lines.append("module load CUDA/12.4.0 GCC/13.2.0 WebProxy/0000")
    lines.append("")
    lines.append("# HF_TOKEN is loaded from a user-provided .env file (never hardcoded; see .env.example).")
    lines.append("# Put a .env next to this script or in the directory you run sbatch from:  HF_TOKEN=hf_xxx")
    lines.append('set -a; [ -f "${SLURM_SUBMIT_DIR:-.}/.env" ] && . "${SLURM_SUBMIT_DIR:-.}/.env"; set +a')
    lines.append('[ -z "${HF_TOKEN:-}" ] && echo "[warn] HF_TOKEN not set and no .env found; gated HF downloads may fail" >&2')
    lines.append("")
    lines.append(f"export FEAT_CACHE={FEAT_CACHE}")
    if holdout_cap:
        # HOLDOUT_CAP is honored via this env var; the drivers do not take --holdout-cap.
        lines.append("export HOLDOUT_CAP=3000")
    lines.append("")
    lines.append("# Two python envs by absolute path: SOAP/MACE -> ase-test-mace; ORB/UMA -> ase-test.")
    lines.append(f"PY_MACE={PY_MACE}")
    lines.append(f"PY_FAIR={PY_FAIR}")
    return "\n".join(lines)


def run_block(model, descriptor, dataset_pkl, pca_var, n_var, splits, out_root, dataset_tag):
    """One descriptor run, routed to the right python env. Uses shell vars for PCA/N."""
    py = "$" + PY_FOR_DESC[descriptor]
    out = (f"{ROOT}/results/{out_root}/{dataset_tag}_pca${{{pca_var}}}_n${{{n_var}}}")
    return (
        f'    echo "=== {model.upper()} {descriptor.upper()} | '
        f'pca=${{{pca_var}}} n_train=${{{n_var}}} ==="\n'
        f"    {py} {ROOT}/src/{model}_regression.py \\\n"
        f"        --dataset {dataset_pkl} \\\n"
        f"        --pca-components ${{{pca_var}}} \\\n"
        f"        --n-train ${{{n_var}}} \\\n"
        f"        --n-splits {splits} \\\n"
        f"        --descriptor {descriptor} \\\n"
        f"        --output-dir {out} \\\n"
        f"        --device cuda\n"
    )


def make_sweep(job_name, walltime, model, descriptors, dataset_pkl, dataset_tag,
               splits, pca_list, n_list, holdout_cap):
    """PCA x N double loop, one or more descriptors per (PCA,N) cell."""
    out_root = OUT_ROOT[model]
    parts = [header(job_name, walltime), "", preamble(holdout_cap), "", ""]
    pca_str = " ".join(str(p) for p in pca_list)
    n_str = " ".join(str(n) for n in n_list)
    body = []
    body.append(f"for PCA in {pca_str}; do")
    body.append(f"    for NTRAIN in {n_str}; do")
    for d in descriptors:
        body.append(run_block(model, d, dataset_pkl, "PCA", "NTRAIN",
                              splits, out_root, dataset_tag).rstrip("\n"))
        body.append("")
    # drop trailing blank inside the loop
    if body and body[-1] == "":
        body.pop()
    body.append("    done")
    body.append("done")
    body.append("")
    body.append(f'echo "=== {job_name} SWEEP COMPLETE ==="')
    return "\n".join(parts) + "\n".join(body) + "\n"


def make_dgp_per_n(job_name, walltime, descriptors, dataset_pkl, dataset_tag,
                   splits, pca_list, ntrain):
    """DGP per-n script: N fixed, loop PCA only. Arm A only (no holdout cap)."""
    out_root = OUT_ROOT["dgp"]
    parts = [header(job_name, walltime), "", preamble(holdout_cap=False), "", ""]
    pca_str = " ".join(str(p) for p in pca_list)
    body = []
    body.append(f"NTRAIN={ntrain}")
    body.append("")
    body.append(f"for PCA in {pca_str}; do")
    for d in descriptors:
        body.append(run_block("dgp", d, dataset_pkl, "PCA", "NTRAIN",
                              splits, out_root, dataset_tag).rstrip("\n"))
        body.append("")
    if body and body[-1] == "":
        body.pop()
    body.append("done")
    body.append("")
    body.append(f'echo "=== {job_name} SWEEP COMPLETE ==="')
    return "\n".join(parts) + "\n".join(body) + "\n"


def write(name, content):
    path = os.path.join(HERE, name)
    with open(path, "w") as f:
        f.write(content)
    os.chmod(path, 0o755)
    return name


def main():
    written = []

    # ===================== ARM A : dfpt, splits=5, no HOLDOUT_CAP =====================
    A_PCA = [10, 25, 50]
    A_N = [100, 250, 500]
    A_SPLITS = 5

    # GP: 2 files
    written.append(write("run_gp_mace_dfpt.slurm", make_sweep(
        "gp_mace_dfpt", WALL_GP, "gp", ["mace"], DFPT_PKL, "dfpt",
        A_SPLITS, A_PCA, A_N, holdout_cap=False)))
    written.append(write("run_gp_soau_dfpt.slurm", make_sweep(
        "gp_soau_dfpt", WALL_GP, "gp", ["soap", "orb", "uma"], DFPT_PKL, "dfpt",
        A_SPLITS, A_PCA, A_N, holdout_cap=False)))

    # MTGP: 2 files
    written.append(write("run_mtgp_mace_soap_dfpt.slurm", make_sweep(
        "mtgp_mace_soap_dfpt", WALL_MTGP, "mtgp", ["mace", "soap"], DFPT_PKL, "dfpt",
        A_SPLITS, A_PCA, A_N, holdout_cap=False)))
    written.append(write("run_mtgp_orb_uma_dfpt.slurm", make_sweep(
        "mtgp_orb_uma_dfpt", WALL_MTGP, "mtgp", ["orb", "uma"], DFPT_PKL, "dfpt",
        A_SPLITS, A_PCA, A_N, holdout_cap=False)))

    # DGP: per-n (3 n) x 2 groups = 6 files
    for n in A_N:
        written.append(write(f"run_dgp_mace_soap_dfpt_n{n}.slurm", make_dgp_per_n(
            f"dgp_mace_soap_dfpt_n{n}", WALL_DGP, ["mace", "soap"], DFPT_PKL, "dfpt",
            A_SPLITS, A_PCA, n)))
    for n in A_N:
        written.append(write(f"run_dgp_orb_uma_dfpt_n{n}.slurm", make_dgp_per_n(
            f"dgp_orb_uma_dfpt_n{n}", WALL_DGP, ["orb", "uma"], DFPT_PKL, "dfpt",
            A_SPLITS, A_PCA, n)))

    # ===================== ARM B : pheasy, splits=10, HOLDOUT_CAP=3000 =====================
    B_PCA = [10, 25, 50]
    B_SPLITS = 10

    # GP: 2 files, n loop {100,250,500,1000,2000}
    B_GP_N = [100, 250, 500, 1000, 2000]
    written.append(write("run_gp_mace_pheasy.slurm", make_sweep(
        "gp_mace_pheasy", WALL_GP, "gp", ["mace"], PHEASY_PKL, "pheasy",
        B_SPLITS, B_PCA, B_GP_N, holdout_cap=True)))
    written.append(write("run_gp_soau_pheasy.slurm", make_sweep(
        "gp_soau_pheasy", WALL_GP, "gp", ["soap", "orb", "uma"], PHEASY_PKL, "pheasy",
        B_SPLITS, B_PCA, B_GP_N, holdout_cap=True)))

    # MTGP: 2 files, n loop {100,250,500,1000} (MTGP capped at 1000)
    B_MTGP_N = [100, 250, 500, 1000]
    written.append(write("run_mtgp_mace_soap_pheasy.slurm", make_sweep(
        "mtgp_mace_soap_pheasy", WALL_MTGP, "mtgp", ["mace", "soap"], PHEASY_PKL, "pheasy",
        B_SPLITS, B_PCA, B_MTGP_N, holdout_cap=True)))
    written.append(write("run_mtgp_orb_uma_pheasy.slurm", make_sweep(
        "mtgp_orb_uma_pheasy", WALL_MTGP, "mtgp", ["orb", "uma"], PHEASY_PKL, "pheasy",
        B_SPLITS, B_PCA, B_MTGP_N, holdout_cap=True)))

    # NO DGP for Arm B.

    for name in written:
        print(name)
    print(f"\nTotal: {len(written)} files")


if __name__ == "__main__":
    main()
