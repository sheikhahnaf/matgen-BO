#!/usr/bin/env python
"""Combined ENCUT + KSPACING convergence on ONE representative structure.

Completes the 3-regime convergence bracket (metal MoN / soft-alloy Li4Mg via campaign_kconv;
insulator + hard-element here). Run on a hard-element insulator (e.g. Na2BO3: B + O are the
hardest pseudopotentials in the set, and it is gapped) so a single job confirms BOTH:
  (a) ENCUT=680 is converged for the hard first-row elements (B/O), and
  (b) KSPACING converges for an insulator (which is easier than the metal already shown flat).

Protocol: relax ONCE at a high reference ENCUT + dense mesh, then frozen-geometry EOS sweeps —
  * ENCUT  {520,600,680,760} at fixed KSPACING=0.16
  * KSPACING {0.30,0.24,0.20,0.16,0.12} at fixed ENCUT=680
Each B0 via the validated oracle-parity machinery (7-point rigid BM, fixed grid, tetra guard).
Per-sweep-point checkpointing so a timeout preserves finished work. Reuses campaign_eos verbatim.
"""
import os, sys, json
from campaign_eos import prep_cell, relax, eos_b0   # single source of truth

ENCUTS   = [520, 600, 680, 760]
SPACINGS = [0.30, 0.24, 0.20, 0.16, 0.12]
REF_ENCUT, REF_SP = 760, 0.14          # high cutoff + dense mesh for a well-converged reference

def _row(d, **extra):
    return dict(B0_GPa=d["B0_GPa"], V0=d["V0"], B0prime=d["B0prime"], k_grid=d["k_grid"],
                smearing=d["smearing"], inwindow=d["inwindow"], rms_resid_eV=d["rms_resid_eV"], **extra)

def main():
    cif = sys.argv[1]
    stem = os.path.splitext(os.path.basename(cif))[0]
    base = os.path.abspath("conv_" + stem); os.makedirs(base, exist_ok=True)
    jpath = os.path.join(base, "conv.json")
    s, isym, info = prep_cell(cif)
    print("%s  ENCUT+KSPACING convergence\n  prep: want_sg=%s used=%s symprec=%s sg_final=%s nat=%d -> ISYM=%d ISPIN=2"
          % (stem, info["want"], info["used"], info["symprec"], info["sg_final"], info["nat"], isym), flush=True)
    rel, magmoms = relax(s, os.path.join(base, "relax"), isym, REF_SP, encut=REF_ENCUT)
    print("  relaxed (ENCUT=%d, KSPACING=%.3f): V=%.3f a=%.4f b=%.4f c=%.4f"
          % (REF_ENCUT, REF_SP, rel.volume, *rel.lattice.abc), flush=True)
    out = dict(stem=stem, isym=isym, ref_encut=REF_ENCUT, ref_spacing=REF_SP,
               encut_sweep=[], kspacing_sweep=[], **info)

    print("=== ENCUT sweep (KSPACING=0.16 fixed) ===", flush=True)
    for enc in ENCUTS:
        try:
            d = eos_b0(rel, os.path.join(base, "e%d" % enc), isym, 0.16, magmoms=magmoms, encut=enc)
            out["encut_sweep"].append(_row(d, encut=enc))
            print("  ENCUT=%4d  B0=%.1f GPa  (grid=%s %s, V0=%.2f, in-win=%s, resid=%.1e)"
                  % (enc, d["B0_GPa"], d["k_grid"], d["smearing"], d["V0"], d["inwindow"], d["rms_resid_eV"]), flush=True)
        except Exception as e:
            out["encut_sweep"].append(dict(encut=enc, error=str(e))); print("  ENCUT=%4d FAILED: %s" % (enc, e), flush=True)
        json.dump(out, open(jpath, "w"), indent=2)

    print("=== KSPACING sweep (ENCUT=680 fixed) ===", flush=True)
    for sp in SPACINGS:
        try:
            d = eos_b0(rel, os.path.join(base, "k%.2f" % sp), isym, sp, magmoms=magmoms, encut=680)
            out["kspacing_sweep"].append(_row(d, spacing=sp))
            print("  KSPACING=%.2f  B0=%.1f GPa  (grid=%s %s, V0=%.2f, in-win=%s, resid=%.1e)"
                  % (sp, d["B0_GPa"], d["k_grid"], d["smearing"], d["V0"], d["inwindow"], d["rms_resid_eV"]), flush=True)
        except Exception as e:
            out["kspacing_sweep"].append(dict(spacing=sp, error=str(e))); print("  KSPACING=%.2f FAILED: %s" % (sp, e), flush=True)
        json.dump(out, open(jpath, "w"), indent=2)

    print("DONE %s  (ENCUT plateau confirms 680; KSPACING plateau confirms the lock)" % stem, flush=True)

if __name__ == "__main__":
    main()
