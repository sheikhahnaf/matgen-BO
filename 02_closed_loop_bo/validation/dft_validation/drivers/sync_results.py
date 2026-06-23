#!/usr/bin/env python
"""Auto-sync the DFT campaign results into RESULTS_AUTO.md and the vasp-dft skill.

Idempotent; meant to run on a periodic loop. Pulls every result JSON from FASTER (K0/phonon/
kconv/conv), regenerates a tables-only RESULTS_AUTO.md (so a bug here can never mangle the curated
insights.md), and copies insights.md + RESULTS_AUTO.md into the skill so the skill is always current.
Prints a one-line summary (counts) so a monitor can emit only when something changed.
"""
import os, json, glob, shutil, subprocess

LOCAL = "/Volumes/SSD1_SMAAA/matinvent-bo/dft_validation"
RES = os.path.join(LOCAL, "results"); FAS = os.path.join(RES, "faster")
SKILL = "/Users/alvi/.claude/skills/vasp-dft/references"
REMOTE = "faster:/scratch/user/ahnafalvi/dft_validation/"

def pull():
    os.makedirs(FAS, exist_ok=True)
    subprocess.run(["rsync", "-am", "--timeout=90", "-e", "ssh -o ConnectTimeout=30",
        "--include=*/", "--include=K0.json", "--include=phonon.json",
        "--include=kconv.json", "--include=conv.json", "--exclude=*", REMOTE, FAS],
        check=False, capture_output=True)

def load(pat):
    out = []
    for f in sorted(glob.glob(os.path.join(FAS, pat))):
        try:
            out.append((os.path.basename(os.path.dirname(f)), json.load(open(f))))
        except Exception:
            pass
    return out

def k0_table():
    rows = load("eos_*/K0.json")
    L = ["## K0 — bulk modulus (%d/29 done)" % len(rows), "",
         "| structure | K0 (GPa) | V0 (A^3) | ISYM | |M| | in-win | smearing |",
         "|---|---|---|---|---|---|---|"]
    for stem, d in sorted(rows, key=lambda r: -r[1].get("B0_GPa", 0)):
        mag = d.get("magmoms"); Mt = sum(abs(m) for m in mag) if mag else 0.0
        L.append("| %s | %.1f | %.2f | %s | %.2f | %s | %s |" % (
            stem.replace("eos_", ""), d["B0_GPa"], d["V0"], d["isym"], Mt, d["inwindow"], d.get("smearing", "")))
    return "\n".join(L), len(rows)

def phonon_table():
    rows = load("ph_*/phonon.json")
    L = ["## Phonons — Cv(300K) / dynamical stability (%d done)" % len(rows), "",
         "| structure | supercell | Cv300 (J/g/K) | min_freq (THz) | n_imag | stable |",
         "|---|---|---|---|---|---|"]
    for stem, d in rows:
        if "cv300_J_per_gK" in d:
            L.append("| %s | %s | %.4f | %.3f | %d | %s |" % (
                stem.replace("ph_", ""), d.get("supercell"), d["cv300_J_per_gK"],
                d.get("min_freq_THz", 0), d.get("n_imaginary", 0), d.get("dynamically_stable")))
        else:
            L.append("| %s | %s | FAILED (%s) |  |  |  |" % (stem.replace("ph_", ""), d.get("supercell"), d.get("failed")))
    return "\n".join(L), len(rows)

def conv_tables():
    L = []; n = 0
    for stem, d in load("kconv_*/kconv.json"):
        n += 1
        L.append("### k-convergence: %s (ISYM=%s)" % (stem.replace("kconv_", ""), d.get("isym")))
        L.append("| KSPACING | B0 (GPa) | grid | smearing |"); L.append("|---|---|---|---|")
        for r in d.get("table", []):
            if "B0_GPa" in r:
                L.append("| %.2f | %.1f | %s | %s |" % (r["spacing"], r["B0_GPa"], r.get("k_grid"), r.get("smearing")))
    for stem, d in load("conv_*/conv.json"):
        n += 1
        L.append("### ENCUT+KSPACING conv: %s (ISYM=%s)" % (stem.replace("conv_", ""), d.get("isym")))
        if d.get("encut_sweep"):
            L.append("ENCUT sweep (KSPACING=0.16): " + ", ".join(
                "%d->%.1f" % (r["encut"], r["B0_GPa"]) for r in d["encut_sweep"] if "B0_GPa" in r))
        if d.get("kspacing_sweep"):
            L.append("KSPACING sweep (ENCUT=680): " + ", ".join(
                "%.2f->%.1f" % (r["spacing"], r["B0_GPa"]) for r in d["kspacing_sweep"] if "B0_GPa" in r))
    return "\n".join(L), n

def main():
    pull()
    k0, nk = k0_table(); ph, npg = phonon_table(); cv, nc = conv_tables()
    md = ["# DFT validation — AUTO-SYNCED results", "",
          "_Regenerated each sync from FASTER result JSONs. Tables only — do not hand-edit; "
          "curated narrative lives in `insights.md`._", "", k0, "", ph, "", "## Convergence study", "", cv, ""]
    open(os.path.join(RES, "RESULTS_AUTO.md"), "w").write("\n".join(md))
    # keep the skill current
    for src, dst in [(os.path.join(LOCAL, "insights.md"), "campaign-insights.md"),
                     (os.path.join(RES, "RESULTS_AUTO.md"), "campaign-results-auto.md")]:
        if os.path.exists(src):
            shutil.copy(src, os.path.join(SKILL, dst))
    print("SYNC k0=%d phonon=%d conv=%d -> RESULTS_AUTO.md + skill" % (nk, npg, nc))

if __name__ == "__main__":
    main()
