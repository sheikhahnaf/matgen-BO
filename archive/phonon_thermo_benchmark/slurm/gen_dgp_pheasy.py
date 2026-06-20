"""Generate the 8 ADDITIVE pheasy-DGP SLURM scripts (2 envs x n{100,250,500,1000}).

Mirrors the MTGP pheasy arm so Arm B becomes a full GP/MTGP/DGP set. Stock DGP architecture
(same as the DFPT DGP / paper) for cross-arm comparability. New filenames + new
results/dgp/pheasy_* output dirs => purely additive; nothing existing is overwritten.
Node-local /tmp feature cache (inode-safe, like the pheasy GP/MTGP scripts).
"""
from pathlib import Path

ROOT = "/scratch/user/u.sa119259/phonon_thermo_benchmark"
OUTDIR = Path("/Volumes/SSD1_SMAAA/matinvent-bo/phonon_thermo_benchmark/slurm")

ENVS = {
    "mace_soap": ("PY_MACE", ["mace", "soap"]),
    "orb_uma":   ("PY_FAIR", ["orb", "uma"]),
}
WALLTIME = {100: "10:00:00", 250: "14:00:00", 500: "24:00:00", 1000: "40:00:00"}

HEADER = """#!/bin/bash
#SBATCH --job-name=dgp_{env}_pheasy_n{n}
#SBATCH --time={wall}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=156192594849
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err


module purge
module load CUDA/12.4.0 GCC/13.2.0 WebProxy/0000

# HF_TOKEN from user-provided .env (never hardcoded).
set -a; [ -f "${{SLURM_SUBMIT_DIR:-.}}/.env" ] && . "${{SLURM_SUBMIT_DIR:-.}}/.env"; set +a
[ -z "${{HF_TOKEN:-}}" ] && echo "[warn] HF_TOKEN not set and no .env found; gated HF downloads may fail" >&2

# Node-local feature cache: pheasy has ~11.8k structures; a persistent per-structure $SCRATCH
# cache would risk the 250k-inode quota. /tmp is node-local, huge inode budget, auto-cleaned;
# features are bit-identical, reused across this job's PCA loop. (Same policy as pheasy GP/MTGP.)
export FEAT_CACHE=/tmp/feat_cache_${{SLURM_JOB_ID}}
export HOLDOUT_CAP=3000

PY_MACE=/scratch/user/u.sa119259/envs/ase-test-mace/bin/python
PY_FAIR=/scratch/user/u.sa119259/envs/ase-test/bin/python

NTRAIN={n}
"""

CELL = """    echo "=== DGP {DESC} | pca=${{PCA}} n_train=${{NTRAIN}} ==="
    ${py} {root}/src/dgp_regression.py \\
        --dataset {root}/data/pheasy_phonon_thermo.pkl \\
        --pca-components ${{PCA}} \\
        --n-train ${{NTRAIN}} \\
        --n-splits 10 \\
        --descriptor {desc} \\
        --output-dir {root}/results/dgp/pheasy_pca${{PCA}}_n${{NTRAIN}} \\
        --device cuda
"""

written = []
for env, (py, descs) in ENVS.items():
    for n in (100, 250, 500, 1000):
        body = HEADER.format(env=env, n=n, wall=WALLTIME[n])
        body += "\nfor PCA in 10 25 50; do\n"
        for desc in descs:
            body += CELL.format(DESC=desc.upper(), py=py, desc=desc, root=ROOT)
        body += "done\n\n"
        body += f'echo "=== dgp_{env}_pheasy_n{n} SWEEP COMPLETE ==="\n'
        f = OUTDIR / f"run_dgp_{env}_pheasy_n{n}.slurm"
        f.write_text(body)
        written.append(f.name)

print(f"wrote {len(written)} scripts:")
for w in sorted(written):
    print("  ", w)
