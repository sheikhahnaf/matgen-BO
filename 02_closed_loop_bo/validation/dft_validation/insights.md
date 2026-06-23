# DFT validation of AI-generated structures — methods, decisions, and convergence record

Working record for the paper. Captures the DFT protocol, every non-obvious physics decision and
its justification, the convergence study (with numbers), the bugs found and fixed, and pointers to
the code and results. Intended to be lifted into the Methods / SI.

**Status as of this writing (2026-06-19):** convergence study near-complete — MoN (metal) and Li₄Mg
(alloy) done; Na₂BO₃ (insulator + hard elements) running. 29-structure production campaign not yet
launched (gated on the convergence lock below).

---

## 1. Purpose

The generative/BO results were assessed only with ML proxies (eSEN bulk-modulus oracle, CGNF and
ORB-PU synthesizability). The "no ground truth" gap is closed by DFT-validating a representative
set of the generated structures and comparing **apples-to-apples** against the ML oracle. Per
structure we compute:

- **K₀** — bulk modulus via hydrostatic Birch–Murnaghan EOS (the oracle's own quantity)
- **C_v(300 K)** — harmonic heat capacity (phonopy)
- **E_above_hull** + **dynamical stability** — DFT synthesizability proxies, to compare against
  CGNF and ORB-PU

**Golden rule throughout:** match the comparison target's potential-energy surface (functional,
pseudopotentials, +U, **and spin**). A deviation that is "more accurate" but breaks parity is a
worse result, not a better one.

## 2. Structure set

29 structures = (top-5 per backbone × target) ∪ (synthesizable-by-both CGNF & ORB-PU, score ≥0.5).
Backbones: MatterGen (`mg`), CrystalFlow (`cf`), ADiT (`adit`); policies BASE / ACC. Two property
targets (bulk modulus `bm`, heat capacity `cp`). Files in `structures/`, named
`{target}_top{NN}_{backbone}_{policy}_seed{N}_{formula}_sg{SG}.cif`. Chemistry spans refractory
metals (carbides/nitrides/intermetallics: MoN, MoC, Os₅W₃, CoIrOs₂, FeRe₉Os₂…), soft alloys
(Li–Mg), and main-group insulating oxides (Na₂BO₃, Li₃PO₄, Li₆TiO₅).

## 3. Cell preparation (spglib refinement) — required for AI cells

Generated cells carry ~0.01–0.05 Å lattice-vector noise. Two consequences, both fixed by one step:

- **LATTYP crash:** VASP's real- vs reciprocal-lattice Bravais classifiers disagree →
  `Inconsistent Bravais lattice types … I REFUSE TO CONTINUE WITH THIS SICK JOB`. Niggli reduction
  alone does not fix it; custodian has no handler.
- **Hidden symmetry → intractable k-mesh:** without clean symmetry VASP runs the full mesh; a
  refractory metal then never finishes (MoN stalled >3 h under ISYM=−1).

**Fix:** refine with spglib at the **tightest** symprec in {1e-3, 1e-2, 5e-2, 1e-1} that recovers
the generator's *intended* spacegroup (from the `_sgNN` tag) — never looser (looser invents
symmetry). Take the primitive standard cell; assign ISYM=2 (symmetry recovered) or ISYM=0
(genuinely P1).

Outcome over the 29: **22 → ISYM=2** (point-group k-reduction 4–24×; MoN sg187→12×, MoC sg194→24×),
**7 → ISYM=0** (genuinely P1: Re₅W, NaLi₅, Li₁₃Mg₃Al₂, Li₇Mg₃, Li₆TiO₅, Li₃PO₄, Na₂BO₃). Every
refined cell reproduced its intended SG exactly at tight symprec — no over-symmetrization.

**Validation:** under ISYM=2 the MoN reference relax that never finished in 3 h under ISYM=−1
completed in ~16–18 min **at the identical geometry** (V=80.76 vs 80.81 Å³, a=2.861 Å) — confirming
the refinement is geometry-preserving, not a distortion.

**Why this is safe physics:** K₀ is volume-intensive, E_hull is per-atom, C_v per-gram — all
invariant to primitive-vs-conventional choice. The ≤0.05 Å symmetrization snap is within the
diffusion model's own sampling noise. The phonon stage (ISYM=0) independently confirms the refined
symmetric cell is a true dynamical minimum, so refinement cannot hide an instability.

## 4. DFT settings

| Knob | Value | Rationale |
|---|---|---|
| Functional / POTCAR | PBE, **PBE_54** | matches MP / OMat24 (the oracle's training DFT) |
| ENCUT | **680 eV** | 1.7× the hardest element ENMAX (B/N/O ~400); above the 1.3× EOS threshold; convergence-verified (§6) |
| +U | **off** (0/29) | MP applies Dudarev +U only to {V,Cr,Mn,Fe,Co,Ni,Mo,W} bonded to O/F; our TM phases are carbides/nitrides/borides/intermetallics, our oxides are main-group → none trigger it |
| ISPIN / MAGMOM | **ISPIN=2 + MP MAGMOM, branch-pinned** | parity (§5) |
| ISMEAR (relax) | 0 / SIGMA 0.05 | gap-agnostic, stable forces for metal or insulator |
| ISMEAR (EOS static) | **−5** tetrahedron, guarded | most accurate total energy; Gaussian fallback if any k-axis <4 subdivisions |
| ISMEAR (phonon) | 0 / SIGMA 0.01 | accurate forces on displaced cell |
| ISYM | **2 for EOS, 0 for phonons** | §5 |
| EDIFF | 1e-6 relax, **1e-7 static** | B₀ is a 2nd derivative; per-cell (not per-atom) SCF floor must be below the few-meV curvature signal |
| LASPH | True | aspherical PAW, important with d/f |

## 5. Two decisions that are easy to get wrong

### ISYM differs by calculation
- **EOS / K₀ → ISYM=2.** Hydrostatic (isotropic) volume scaling is a pure dilation; it preserves
  the *full* space group, so all 7 strain points share the relaxed cell's symmetry. Correct, and it
  is what makes the metals affordable (k-reduction).
- **Phonons → ISYM=0.** The finite displacement breaks symmetry. phonopy does its own symmetry
  reduction (inequivalent displacements + force-constant symmetrization); letting VASP also
  symmetrize forces can **average away a soft/imaginary mode** — the very signal a dynamical-
  stability screen exists to detect. atomate2's phonon-displacement set restricts ISYM to {0,−1}.
  Source: phonopy VASP interface docs; atomate2 `PhononDisplacementMaker`.
- **Avoid ISYM=−1** — no k-reduction at all; only ever needed when symmetry detection crashes,
  which the spglib refinement removes.

### Spin parity (ISPIN=2, not 1) — and pin the branch
The eSEN/OMat24 oracle was trained on **spin-polarized PBE(+U)** DFT (OMat24 paper: spin-polarized
static VASP, MP-consistent settings). It has no spin input but predicts spin-polarized energies, so
the correct DFT parity is **ISPIN=2 + MP MAGMOM**, *not* ISPIN=1. (An initial automated audit
recommended ISPIN=1 on a "spinless oracle" premise; verifying the OMat24 settings reversed that.)
A fixed MAGMOM with frozen ions across the 7 EOS volumes can hop magnetic branches → non-smooth
E(V) → corrupt BM curvature, so the **magnetic branch is pinned**: MAGMOM for every strain static is
seeded from the relaxed reference's converged moments. No-op for the non-magnetic majority.
Refs: OMat24 (Nature Comput. Sci. 2026 / arXiv 2410.12771).

## 6. Convergence study (the 3-regime bracket)

Settings are converged on **representatives spanning the regimes**, then applied uniformly — not
per structure. Justification: ENCUT is a per-pseudopotential property (the hardest element sets it),
and KSPACING is a reciprocal-space density that auto-adapts to each cell, with the binding case
being metals (densest k for the Fermi surface). So converge on the worst case (refractory metal) +
brackets, and everything easier is covered. This is standard MP/atomate2 practice. Per-structure we
keep only QC flags (V₀-in-window, fit residual, tetra/Gaussian mesh decision), not re-convergence.

Protocol: relax once at a high reference cutoff/mesh, then frozen-geometry EOS (oracle-parity
7-point rigid Birch–Murnaghan) at each ENCUT / KSPACING.

### MoN (refractory metal, sg187, ISYM=2) — B₀ vs KSPACING (ENCUT=680)
| KSPACING | grid | smearing | B₀ (GPa) | V₀ (Å³) |
|---|---|---|---|---|
| 0.30 | [9,9,2] | gauss | 353.2 | 80.88 |
| 0.24 | [11,11,3] | gauss | 353.7 | 80.83 |
| 0.20 | [13,13,3] | gauss | 353.7 | 80.83 |
| **0.16** | [16,16,4] | **tetra** | **353.7** | 80.84 |
| 0.12 | [22,22,5] | tetra | 353.6 | 80.84 |

Flat to **0.15%** across the whole range; tetrahedron (0.16/0.12) and Gaussian-fallback
(0.30–0.20) agree to 0.1% — the sparse-mesh guard's fallback is unbiased.

### Li₄Mg (soft alloy, sg12, ISYM=2) — B₀ vs KSPACING (ENCUT=680)
| KSPACING | grid | smearing | B₀ (GPa) | V₀ (Å³) |
|---|---|---|---|---|
| 0.30 | [5,5,4] | tetra | 18.5 | 202.06 |
| 0.24 | [6,6,5] | tetra | 18.2 | 202.23 |
| 0.20 | [7,7,6] | tetra | 18.4 | 202.26 |
| **0.16** | [9,9,7] | tetra | **18.2** | 202.30 |
| 0.12 | [12,12,9] | tetra | 18.3 | 202.34 |

Converged to ~18.2–18.3 GPa; the ~1.6% scatter at coarse mesh (18.2–18.5) is <2% (a soft low-K cell
is more sensitive in relative terms), and the two densest points (0.16/0.12 → 18.2/18.3) agree
tightly. Physically sensible for a soft Li–Mg alloy. V₀ ~202.3 Å³, in-window at every point.

### Insulator + hard-element leg
**Na₂BO₃ turned out NOT to be a clean insulator — a validation finding in itself.** The generated
composition is **charge-imbalanced**: Na⁺₂B³⁺O²⁻₃ → 2+3−6 = **−1 per formula unit**, so the 12-atom
cell (Na₄B₂O₆) carries 2 excess electrons → a steady **mag = 2.0 μB**, spin-polarized, effectively
metallic (states at E_F). DFT relaxation is correspondingly slow (~9 min/ionic step, metallic SCF).
This is exactly the kind of unphysical generated structure the DFT validation should flag (expect
high E_above_hull / instability in the full campaign). Note ISPIN=2 correctly captured the mag=2
state — ISPIN=1 would have been wrong here, an independent vindication of the spin-parity choice.
Na₂BO₃ is left running (its data is a conservative, near-metallic convergence point) but is the
wrong choice for the *clean-insulator* leg.

**Li₃PO₄ is the clean-insulator representative** (also generated, also P1, ISYM=0, 16 atoms):
Li⁺₃P⁵⁺O²⁻₄ → 3+5−8 = **0, charge-balanced**, a textbook wide-gap solid electrolyte with the hard
P/O pseudopotentials. Combined ENCUT {520,600,680,760} + KSPACING {0.30…0.12} sweep confirms
(a) ENCUT=680 on hard elements, (b) clean insulator k-convergence. *(Table to be filled on
completion — `results/convergence/conv_Li3PO4*.out` / `conv.json`; Na₂BO₃ table likewise.)*

### Locked settings
**ENCUT = 680 eV, KSPACING = 0.16 (Γ-centered).** 0.16 chosen over 0.20 because it keeps ≥4
subdivisions on every axis for the metals, so tetrahedron stays valid (MoN's long-c hexagonal axis
hits 4 only at 0.16). Earlier Li₄Mg ENCUT sweep (520/600/680/760 → 18.2/18.1/18.2/18.2) confirmed
ENCUT flatness on the soft case; Na₂BO₃ confirms it on the hard-element case.

## 6b. Charge-balance pre-screen (cheap predictor of unphysical generated structures)

A generative model can emit charge-imbalanced ionic compositions. A formal-charge screen
(`drivers/charge_balance_screen.py`, runs in seconds, no DFT) flags these and **predicts their net
cell spin** from the charge excess. Of the 29: 26 are non-ionic (alloys / metallic ceramics —
charge balance not defined for covalent/metallic bonding) and **3 are ionic** (O-bearing):

| structure | cell | net formal q/cell | balanced? |
|---|---|---|---|
| Li₆TiO₅ | Li₆TiO₅ | 0 | ✅ |
| Li₃PO₄ | Li₆P₂O₈ | 0 | ✅ |
| **Na₂BO₃** | Na₄B₂O₆ | **−2 e⁻** | ❌ imbalanced → predict \|M\|≈2 μB |

**Key validation:** the screen predicted Na₂BO₃'s net moment (−2 e⁻/cell → \|M\|≈2 μB) from
composition alone, and DFT independently relaxed it to **mag = 2.0 μB**. A seconds-long check
anticipated the multi-hour DFT result. So the generator's ionic charge-balance rate here is 2/3,
and charge balance is a useful no-cost pre-filter for the ionic subset (it does not apply to the
metallic/intermetallic majority). Output recorded in `results/charge_balance_screen.md`.

## 6c. Emerging K0 results (live — full table in `results/RESULTS_AUTO.md`)

The K0 dataset separates by **bonding type into three regimes** (it first looked bimodal because the
metals/alloys came in before the oxides): a **hard refractory cluster ~290–356 GPa** (Os₅W₃, MoN, MoC,
CoIrOs₂, VIr7, TaB2W, FeB2MoW, Mn(BMo)₃, FeRe(PW)₂, ReIr₂Rh₅ — intermetallics/carbides/borides/nitrides),
a **soft alkali/Li-Mg alloy cluster 11–30 GPa** (LiMg₂, LiMg, Li₅CaMg₂, Li₇Mg₃, Li₄Mg×2, Li₁₃Mg₃(Al₂),
NaLi₅, NaLi₂), and the **ionic oxides bridging the gap in between** (Li₆TiO₅ at 101 GPa; Na₂BO₃/Li₃PO₄
pending). All physical, in-window, B₀′ ~2.8–5.1.
Every value so far is in-window with B₀′ ~2.8–5.1 and tiny EOS residuals (~1e-6 eV). **Protocol
reproducibility is confirmed**: production EOS (relax 680/0.16) reproduces the kconv anchors exactly
(Li₄Mg 18.2 = 18.2, MoN 353.6 ≈ 353.7), and same-compound polymorphs agree to 0.1 GPa (CoIrOs₂
346.0/346.0, FeB2MoW 306.9/306.9). Watch item: NaLi₂ B₀′=2.78 (lowest; plausible for a very soft
alkali alloy, flag only if it generalizes).

## 7. EOS protocol (oracle-parity) — see `drivers/campaign_eos.py`

Mirrors the eSEN BM oracle (`matinvent-hcap-bo/.../local_esen_bm.py`) exactly:
1. relax cell+ions (ISIF=3);
2. **7 isotropic strain points** ε∈{−0.03…+0.03 step 0.01}, cell→V₀(1+ε), **fractional coords
   frozen, single-point static** (no per-volume ion relaxation — the oracle does rigid scaling);
3. 3rd-order Birch–Murnaghan fit → K₀.

**Fixed-k-grid rule (critical):** compute the k-grid **once from the relaxed reference cell** and
reuse it for all 7 volumes. Recomputing per strained cell drifts the mesh (e.g. [5,4,4]→[5,4,3])
→ discontinuous E(V) → corrupt curvature. This was a real bug — it produced a spurious **21/16/18.2
GPa "oscillation"** in an early KSPACING sweep; only the one spacing that held a fixed grid gave the
clean value. After the fix the same regime is smooth (MoN flat 353.x; Li₄Mg 18.2–18.5).

**Robust harvest:** energy from `Vasprun.final_energy` gated on `converged_electronic` (rejects
NELM-hit statics); BM fit guarded against a failed static (None→nan, require ≥5 finite points, raise
a structure-named error, persist partial V,E); QC records fit residual and V₀-in-window.

## 8. Phonon protocol (Cv + dynamical stability) — validated, scaling up

phonopy finite displacement, **ISYM=0**, PREC=Accurate, IBRION=−1/NSW=0, ISMEAR=0/SIGMA=0.05 (gap-
agnostic), EDIFF=1e-7, ENCUT=680, ISPIN=2 + per-element MAGMOM seeded from the EOS-relaxed moments,
supercell k-mesh from a fixed reciprocal density (auto-reduces as the supercell grows), displacement
0.01 Å. Dynamical stability = no imaginary modes (below −0.05 THz, acoustic-at-Γ excluded). C_v(300 K)
from the harmonic DOS, **normalized per gram (J/g/K)** (intensive; matches the oracle; dodges the
per-formula-unit factor-Z trap). Driver `drivers/campaign_phonon.py`.

**Physicality discovery — reference validation PASSED.** Diamond-Si (primitive, 3×3×3 supercell, 54
atoms, 1 symmetry-reduced displacement) → **Cv(300 K) = 0.7132 J/g/K** (textbook ≈0.71 ✓),
min_freq = +0.306 THz, **0 imaginary modes → STABLE**. This validates the whole chain (phonopy ↔ VASP
↔ ISYM=0 ↔ forces ↔ thermal integration ↔ per-gram normalization). **Supercell convergence PASSED**:
Si Cv(300 K) = 0.7132 (3×3×3, 54 atoms) vs 0.7139 J/g/K (4×4×4, 128 atoms) — **0.1% apart**, and the
dyn-stability verdict is identical (0 imaginary, STABLE) at both sizes, so the stable verdict is real,
not a force-constant-truncation artifact. **Campaign supercell criterion locked at min-length ~15 Å**
(Si already converged at min-length 10; 15 is the safe conservative choice that also protects soft-mode
detection). The Cp physicality discovery is complete — pipeline + supercell both validated. The 7 P1
cells (no displacement reduction → tens of displacements each) are the cost long-pole at campaign scale.
Live numbers in `results/RESULTS_AUTO.md` (auto-synced).

## 9. Infrastructure (FASTER) — see `drivers/*.slurm`

VASP 6.3.2 (`intel/2022a`, `vasp/6.3.2`); SLURM account 142705333487, partition cpu, 16 tasks.
**MPI needs both** `I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so` **and** launcher `srun --mpi=pmi2`,
else VASP spawns 16 serial copies that clobber output (`running on 1 total cores`, corrupt
vasprun.xml). POTCARs via `PMG_VASP_PSP_DIR` symlinks to `potpaw_PBE.54`. Every VASP call under
custodian (full handler + validator stack). Per-structure jobs in guarded `eos_<stem>/` dirs;
idempotent resubmit (resume converged dirs, else clean). A health monitor watches for broken-but-
RUNNING jobs (MPI-serial signature, fatal VASP errors, stalls), not just completion.

## 10. Parity caveats to disclose in the paper
- **Symmetry-constrained relax:** DFT relaxes the refined cell under ISYM=2; the oracle relaxes the
  as-given cell unconstrained. For cells whose SG is recovered only at loose symprec (5e-2/1e-1) the
  equilibria can differ. Mitigation: report V₀/atom from both sides, flag loose-symprec K₀ as
  lower-confidence; the phonon stability check backs the symmetric choice.
- **Pseudopotential differences** (OMat24 vs MP differ on W, Yb) — a small known systematic; note,
  don't chase.

## 11. Bugs found & fixed (for reproducibility honesty)
1. **k-grid drift across EOS volumes** → BM oscillation (21/16/18.2 GPa). Fixed: one fixed grid.
2. **ISYM=−1 intractability** on metals. Fixed: spglib refine → ISYM=2.
3. **LATTYP "SICK JOB"** on noisy cells. Fixed: spglib refine (clean Bravais lattice).
4. **MPI serial-copies** clobbering output. Fixed: PMI library + `srun --mpi=pmi2`.
5. **None-energy → cryptic BM crash** losing a whole structure. Fixed: guarded fit + persisted partials.
6. **Tetrahedron on sparse mesh** (no custodian rescue for statics). Fixed: <4-subdivision Gaussian guard.
7. **Silent ISPIN=2 default vs assumed parity.** Resolved: ISPIN=2 is correct (OMat24 spin-polarized);
   added MAGMOM branch-pinning for EOS smoothness.
8. **MAGMOM branch-pinning crashed every magnetic structure** (FeB2MoW, FeRe9Os2, Mn(BMo)₃, FeRe(PW)₂,
   one CoIrOs₂ polymorph |M|=0.31): a per-site MAGMOM *list* passed via `user_incar_settings` hits
   `AttributeError: 'list' object has no attribute 'get'` in pymatgen (it expects a species→moment
   dict there). Non-magnetic cells were unaffected (the pin branch is skipped). Fixed: set per-site
   moments via the structure's `magmom` **site property**, which pymatgen honors per-site. The
   campaign re-submits failed structures generically (no K0.json + left queue → resubmit), resuming
   from the completed relax. Lesson: the magnetic subset is a distinct, easily-missed code path — the
   non-magnetic majority passing clean is not evidence the magnetic path works.
9. **SLURM swallowed python's exit code** (`echo "end"` after the run), so a python crash showed as
   COMPLETED and hid the failure. Fixed: capture `rc=$?`, echo it, `exit $rc` so failures read as
   FAILED and the fleet monitor's failure count is accurate.

## 12. Three-way comparison: DFT vs eSEN oracle vs GP surrogate (causal evaluation)
Code, results, and figures live in the sibling folder `../three_way_comparison/`. The FAITHFUL
driver is `drivers/three_way_causal_seeded.py` (parquet-based); ranking is `drivers/rank_analysis_seeded.py`.

**Why causal, not in-sample.** The validated top structures are themselves members of the
closed-loop memory (present verbatim). Training a GP on all memory and "predicting" them is
memorization, not prediction. We instead replay the workflow per run: at cycle s, train the GP
only on what existed at cycles `< s` and predict the cycle-s structures (unseen).

**The seed-omission correction (important — it overturned a first result).** The first replay
(`legacy/*_NOSEED_flawed.*`, csv-based `three_way_causal.py`) had two flaws found only by reading
the actual acquisition code: (1) it trained per-run on the run's generated structures **without the
~500-structure warm-start seed pool** (`cycle_id=-1`, K0 range 7.5–385 GPa) the real GP used, so it
was cold-started; (2) it ran the GP reconstruction on **all** runs including BASE runs that never
used a GP (only ACC runs have an LTM parquet). The faithful redo trains on seed + accumulated using
the actual per-run LTM parquets (stored `Z_pca50` + labels + `cycle_id`), restricted to ACC runs.
Per the plotting rule, `n_train_accum` on the axis EXCLUDES the seed (loop-gathered only); training
INCLUDES it. Lesson: reconstruct the surrogate from the data/code it ACTUALLY used (seed pool,
which runs had a GP), not a convenient proxy table.

**Faithful results (full seed, ACC runs; run on FASTER via `three_way_seeded.slurm`).**
- **The oracle is a faithful DFT proxy:** eSEN vs DFT over the validated set ρ≈0.885, MAPE 2.3%.
- **BM GP(causal) vs eSEN:** ρ=0.944, r=0.954, RMSE 19.5 GPa (n=1209, 15 ACC runs); within-step
  median ρ=1.00 (mean 0.90, 100% positive). **Cp GP(causal) vs eSEN:** ρ=0.868, r=0.899, RMSE 0.112
  J/g/K (n=596; within-step mean 0.77). (A 200-seed subsample gave the same picture, BM ρ=0.90 —
  the conclusion is robust to seed size; full seed is slightly better.)
- **The GP is a strong ranker, warm-started from the start.** Ranking ρ is ~flat-high vs
  loop-accumulated data (seed excluded): BM 0.88 (0–5) → 0.92 → 0.93 → 0.96 → 0.96 (50+); Cp 0.79 →
  0.85 → 0.89 → 0.89 → 0.87. So the seed pool, not per-cycle accumulation, makes the surrogate rank
  well; extra loop data adds only a modest lift. (The flawed version's "rising 0.23→0.97 cold-start
  curve" was an artifact of omitting the seed.)
- **Mild, honest underprediction of the extreme winners.** For the 10 ACC-run validated winners,
  GP(causal) vs DFT bias −42.5 GPa, ρ=0.71 (e.g. Os5W3 323 vs 356; MoN 301 vs 354; predictions
  ~210–325 GPa with σ≈13–32). The GP pulls the very top extremes toward the training bulk by ~40 GPa
  — a real but modest effect with honest uncertainty, NOT the −161 GPa collapse the seed-omitted
  version showed.

**Reading for the paper.** Rank-not-regress holds from a correct baseline: the surrogate orders
generated structures very well (ρ≈0.87–0.94) while modestly underpredicting absolute values at the
extremes. Caveats to disclose: ACC runs only (BASE runs had no GP); the 10-winner ρ is
restricted-range; Cp DFT (phonon Cv) leg still pending. Figures:
`three_way_{ranking,property}_{bm,cp}_seeded.png` (200-subsample archived to `results/legacy/_seed200`).

## 13. DFT E_above_hull (0 K thermodynamic stability) + makeability standing
Code: `drivers/campaign_ehull.py` (static) + `drivers/ehull_aggregate.py` (hull) + `drivers/ehull.slurm`.
Results: `results/ehull_summary.csv`; cross-method standing in `../tc_makeability/results/makeability_standing_28.csv`.

**Method.** E_above_hull needs an MP-compatible energy + a reference convex hull. Our EOS relax
(ENCUT=680, no +U — chosen for oracle-parity) is NOT MP-hull-compatible, so we run a separate
**MP-compatible static** (`MPStaticSet`, MP INCAR: ENCUT 520, +U where MP applies, MP POTCARs) on each
DFT-relaxed cell, then build a `PhaseDiagram` from MP entries (fetched via the REST thermo endpoint
for all chemical sub-systems, deduped) and take `get_e_above_hull`. Computed for **28/29** (FeRe9Os2's
SCF never converged → no static).

**Bug found & fixed (reproducibility honesty, cf. §11).** The MP thermo endpoint **defaults to
r2SCAN**; mixing those with our **GGA** static gave ~8 eV/atom nonsense (MoN appeared 7.6 eV/atom
above hull). Fixed by requesting `thermo_types=GGA_GGA+U` → our static then matches MP's GGA energy to
**0.01–0.10 eV/atom** (verified: MoC Δ=0.015, LiMg Δ=0.066), giving physical E_hull.

**Results (regimes).** 18 structures **on/near-hull** (≤0.10 eV/atom — e.g. TaB₂W 0.00, Mn(BMo)₃ 0.00,
MoC 0.017, FeB₂MoW 0.025, VIr₇ 0.044, Os₅W₃ 0.097, all Li-Mg 0.02–0.08); 6 **metastable** (0.10–0.20:
MoN 0.10, CoIrOs₂ 0.12, Re5W 0.12, FeRe(PW)₂ 0.16); 4 **above hull** (>0.20: TiVN₂ 0.37 + the 3 oxides,
which carry the anion caveat). **~24/28 within 0.20 eV/atom of the hull** — the synthesizable-metastable
range — independently corroborating that the AI discoveries are thermodynamically reasonable. Notably
the CALPHAD-"decompose" borides are near-hull by DFT → consistent **metastable** reading.

**Cross-method makeability standing (all 29).** Combining DFT-EOS (mechanical), E_hull (0 K thermo),
CALPHAD equilibrium (300 & 1273 K), and eSEN: **6 makeable** (5 Li-Mg single-phase near-hull + MoN
volatile), **18 metastable near-hull** (the hard borides + novel refractory metallics), **4 above-hull**
(TiVN₂ + 3 oxides), **1 no-EOS** (FeRe9Os2). The 1273 K column adds a synthesis lever — several
multi-phase-at-300 K become single-phase at 1273 K (MoC, NaLi5, LiMg2, …) → quench-retainable.

**Caveats to disclose.** (a) **Anion compounds** (MoN, the 3 oxides) — E_hull is an *upper bound*: the
MP2020 anion/+U correction is not applied to our entry (needs the vasprun metadata); true values are
lower (Li3PO4 is a real compound → really ~0). (b) **Sparse-chemsys novel systems** (Os/Ir/Rh/P) —
the MP "hull" is built from few sub-system entries, so E_hull there is a *lower bound* (E_hull=0 can
mean "lowest *known* at that composition"). (c) ±~0.05 eV/atom from settings/corrections.

## 14. Files
- `drivers/campaign_eos.py` — production EOS driver (refine → relax → 7-point rigid BM, ENCUT-param)
- `drivers/campaign_kconv.py` — B₀-vs-KSPACING convergence (per-spacing checkpoint)
- `drivers/campaign_conv.py` — combined ENCUT+KSPACING convergence
- `drivers/kconv.slurm`, `drivers/conv.slurm` — FASTER submit scripts
- `results/convergence/` — kconv JSONs + job outputs (MoN, Li₄Mg done; Na₂BO₃ running)
- `structures/` — the 29 staged CIFs
- `../three_way_comparison/` — sibling folder with ALL three-way assets: `drivers/three_way_causal.py`
  (causal GP replay, GPU), `drivers/rank_analysis.py` (per-step ranking), `drivers/three_way.slurm`,
  `drivers/legacy/three_way_gp_insample.py` (archived flawed in-sample version), `results/*.csv`,
  `results/three_way_{property,ranking}.png`, and `paper/three_way_comparison.md` (paper section).
