#!/usr/bin/env python
"""Metal k-convergence — finish the conv-test arm that stalled under ISYM=-1.

Re-runs the B0-vs-KSPACING test on the spglib-REFINED cell with ISYM=2, so it completes and
gives the metal data point needed to lock the campaign KSPACING. Reuses the validated
campaign_eos machinery (prep_cell / relax / eos_b0) verbatim -> k-conv and production EOS share
one code path (oracle-parity 7-point rigid BM, ENCUT=680, ISPIN=2, fixed-grid + tetrahedron).

Hardened (pre-flight audit): per-spacing checkpointing -> a walltime timeout preserves every
finished spacing instead of discarding the whole job.
"""
import os, sys, json
from campaign_eos import prep_cell, relax, eos_b0   # single source of truth

def main():
    cif = sys.argv[1]
    ref_sp = float(sys.argv[2]) if len(sys.argv) > 2 else 0.14   # dense reference-relax mesh
    spacings = [0.30, 0.24, 0.20, 0.16, 0.12]
    stem = os.path.splitext(os.path.basename(cif))[0]
    base = os.path.abspath("kconv_" + stem); os.makedirs(base, exist_ok=True)
    jpath = os.path.join(base, "kconv.json")
    s, isym, info = prep_cell(cif)
    print("%s  metal k-convergence\n  prep: want_sg=%s used=%s symprec=%s sg_final=%s nat=%d -> ISYM=%d ISPIN=2"
          % (stem, info["want"], info["used"], info["symprec"], info["sg_final"], info["nat"], isym), flush=True)
    rel, magmoms = relax(s, os.path.join(base, "relax"), isym, ref_sp)
    print("  relaxed (ref KSPACING=%.3f): V=%.3f a=%.4f b=%.4f c=%.4f" % (ref_sp, rel.volume, *rel.lattice.abc), flush=True)
    print("=== B0 vs KSPACING (ENCUT=680, ISYM=%d) ===" % isym, flush=True)
    out = dict(stem=stem, isym=isym, ref_spacing=ref_sp, table=[], **info)
    for sp in spacings:
        try:
            d = eos_b0(rel, os.path.join(base, "k%.2f" % sp), isym, sp, magmoms=magmoms)
            row = dict(spacing=sp, B0_GPa=d["B0_GPa"], V0=d["V0"], B0prime=d["B0prime"],
                       k_grid=d["k_grid"], smearing=d["smearing"], inwindow=d["inwindow"],
                       rms_resid_eV=d["rms_resid_eV"])
            print("  KSPACING=%.2f  B0=%.1f GPa  (grid=%s %s, V0=%.2f, in-win=%s, resid=%.1e)"
                  % (sp, d["B0_GPa"], d["k_grid"], d["smearing"], d["V0"], d["inwindow"], d["rms_resid_eV"]), flush=True)
        except Exception as e:
            row = dict(spacing=sp, error=str(e))
            print("  KSPACING=%.2f  FAILED: %s" % (sp, e), flush=True)
        out["table"].append(row)
        with open(jpath, "w") as fh:                 # checkpoint after EACH spacing
            json.dump(out, fh, indent=2)
    print("DONE %s  (lock KSPACING where B0 changes <2%%)" % stem, flush=True)

if __name__ == "__main__":
    main()
