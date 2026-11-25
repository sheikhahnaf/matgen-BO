"""Upload the fine-tuned generator checkpoints of the finished revision runs to
the Hugging Face archive, next to those of the original runs.

    finetuned_diffusion/<run name>/        models/final of each finished run
    revision_round1/calibration_pheasy/    Pheasy holdout predictive distributions

Additive and idempotent: a run whose prefix already has files is skipped.
Frozen-generator runs are skipped, because their weights are the public
pretrained checkpoints (verified by the fingerprint in each run log).
Needs REV_ROOT and a write token in HF_TOKEN.
"""
import glob, os
from huggingface_hub import HfApi

REPO = "SheikhAhnaf/apu-synthesizability-checkpoints"
ROOT = os.environ["REV_ROOT"]
api = HfApi(token=os.environ["HF_TOKEN"])
files = api.list_repo_files(REPO, repo_type="model")
have = {f.split("/")[1] for f in files if f.startswith("finetuned_diffusion/")}

todo, skipped = [], 0
for d in sorted(glob.glob(f"{ROOT}/runs/rev1_*")):
    name = os.path.basename(d)
    if "_frozen_gate_" in name:
        continue
    log = os.path.join(d, "main.log")
    final = os.path.join(d, "models", "final")
    if not (os.path.isfile(log) and "RL END" in open(log, errors="replace").read()):
        continue
    if not glob.glob(os.path.join(final, "*.ckpt")):
        continue
    if name in have:
        skipped += 1
        continue
    todo.append((name, final))
print(f"[plan] {len(todo)} to upload, {skipped} already in the archive", flush=True)
for i, (name, final) in enumerate(todo, 1):
    api.upload_folder(folder_path=final, path_in_repo=f"finetuned_diffusion/{name}", repo_id=REPO,
                      repo_type="model", commit_message=f"Add fine-tuned diffusion checkpoint: {name}")
    print(f"[{i}/{len(todo)}] {name} done", flush=True)

cal = f"{ROOT}/calibration/out/pheasy"
if os.path.isdir(cal) and os.environ.get("UPLOAD_PHEASY_CALIBRATION") == "1":
    api.upload_folder(folder_path=cal, path_in_repo="revision_round1/calibration_pheasy", repo_id=REPO,
                      repo_type="model", allow_patterns=["*/calib_preds_*.csv.gz", "*/gp_*_holdout_summary.csv"],
                      commit_message="Add Pheasy GP calibration predictions (revision round 1)")
    print("[pheasy calibration uploaded]", flush=True)
print("[all done]", flush=True)
