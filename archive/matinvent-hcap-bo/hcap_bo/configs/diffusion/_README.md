# Per-backend Hydra-style override fragments

Each `<model>.yaml` here is a *partial* config that gets merged into `configs/hcap_bo.yaml`
at runtime by `scripts/run_phase2_per_backend.slurm`. The merge sets:

```yaml
generation:
  adapter:
    _target_: src.diffusion.RemoteGeneratorAdapter
    env_prefix: ${oc.env:SCRATCH}/envs/mat-zoo-modern
    model_name: <name>
    adapter_kwargs:
      checkpoint: <abs path>
      device: cuda
      ...
    timeout_seconds: 1800
```

so that `phase2_open._make_pool_provider` instantiates the right `RemoteGeneratorAdapter`
and per-cycle generation goes through subprocess + extxyz IPC into the `mat-zoo-modern`
env that has the upstream model installed.

## Submission usage

```bash
# Single model
sbatch --export=ALL,MODEL=crystalflow scripts/run_phase2_per_backend.slurm

# All 7 in parallel
for MODEL in crystalflow crysbfn symmcd flowmm crystalformer atomgpt adit; do
  sbatch --export=ALL,MODEL=$MODEL --job-name=p2_$MODEL scripts/run_phase2_per_backend.slurm
done
```

## Checkpoint locations

Checkpoints land at `$SCRATCH/checkpoints/<model>/...`. Paths in each YAML reference these
via `${oc.env:SCRATCH}` so the same file works from any node. Update the `checkpoint:` field
in each fragment below as new weights are downloaded.
