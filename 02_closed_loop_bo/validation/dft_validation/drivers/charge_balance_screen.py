#!/usr/bin/env python
"""Charge-balance screen on the 29 generated structures.

A generative model can emit compositions that are NOT charge-balanced (e.g. Na2BO3 -> -1/f.u.).
For an IONIC compound that means excess/deficit electrons -> net spin and likely high E_above_hull
/ instability -- exactly what DFT validation should flag. This pre-screens which structures will
show that, and PREDICTS their net cell spin from the formal-charge excess.

Charge balance is only meaningful for ionic compounds (clear anion). So we classify:
  - ionic            : contains O or F (unambiguous anion) -> screen formally
  - alloy            : all-metallic (no nonmetal)          -> N/A (no ionic charges)
  - metallic ceramic : metal + C/N/B/P/Si, no O/F          -> N/A (covalent/metallic, ambiguous)
For ionic ones: net formal charge per cell with the dominant ionic oxidation state of each element;
cross-checked against pymatgen oxi_state_guesses() (searches common states for a neutral combo).
"""
import glob, os
from pymatgen.core import Structure, Composition, Element

DIR = "/Volumes/SSD1_SMAAA/matinvent-bo/dft_validation/structures"
# dominant ionic oxidation state in an O/F context (unambiguous for these simple compounds)
OXI = dict(Li=1, Na=1, K=1, Rb=1, Cs=1, Mg=2, Ca=2, Sr=2, Ba=2, Al=3, B=3, Ga=3,
           P=5, Si=4, Ti=4, N=-3, O=-2, S=-2, F=-1, Cl=-1)

def classify(comp):
    els = {e.symbol for e in comp.elements}
    if els & {"O", "F"}:
        return "ionic"
    if all(Element(e).is_metal for e in els):
        return "alloy"
    return "metallic-ceramic"

def main():
    cifs = sorted(glob.glob(os.path.join(DIR, "*.cif")))
    print("Charge-balance screen — %d structures\n" % len(cifs))
    hdr = "%-50s %-16s %-7s | %-9s %-11s %s" % ("file", "cell formula", "class", "net_q/cell", "balanceable", "flag")
    print(hdr); print("-" * len(hdr))
    flagged = []
    for c in cifs:
        comp = Structure.from_file(c).composition          # full-cell composition
        cls = classify(comp)
        if cls == "ionic":
            netq = sum(OXI.get(e.symbol, 0) * amt for e, amt in comp.items())
            try:
                balanceable = len(comp.oxi_state_guesses(max_sites=-20)) > 0
            except Exception:
                balanceable = None
            bad = (abs(netq) > 0.01) or (balanceable is False)
            flag = ""
            if bad:
                ne = abs(round(netq))
                flag = "IMBALANCED  net=%+d e-/cell -> predict |M|~%d uB" % (round(netq), ne)
                flagged.append((os.path.basename(c), comp.reduced_formula, round(netq), ne))
            print("%-50s %-16s %-7s | %+9.1f %-11s %s"
                  % (os.path.basename(c)[:50], comp.formula.replace(" ", ""), cls, netq, str(balanceable), flag))
        else:
            print("%-50s %-16s %-7s | %-9s %-11s %s"
                  % (os.path.basename(c)[:50], comp.formula.replace(" ", ""), cls, "N/A", "N/A", "(not ionic — charge balance not defined)"))
    print("\n=== FLAGGED (charge-imbalanced ionic) ===")
    if not flagged:
        print("  none")
    for fn, rf, nq, ne in flagged:
        print("  %-40s  %-10s  net %+d e-/cell  -> predicted net spin ~%d uB" % (rf, fn[:40], nq, ne))
    print("\nNote: predicted |M| should match the DFT cell moment for the flagged ionic structures.")

if __name__ == "__main__":
    main()
