"""Tests for score_structures.py (offline-only subset)."""
import os

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# parse_tag tests
# ---------------------------------------------------------------------------

def test_parse_tag_bm_adit():
    from apu_synthesizability.score_structures import parse_tag
    t = parse_tag("bm_top11_adit_ACC_seed17_TaB2W_sg38.cif")
    assert t.target == "bm"
    assert t.rank == 11
    assert t.backbone == "adit"
    assert t.backbone_name == "ADiT"
    assert t.policy == "ACC"
    assert t.seed == 17
    assert t.formula == "TaB2W"
    assert t.spacegroup == 38


def test_parse_tag_cp_mg_base():
    from apu_synthesizability.score_structures import parse_tag
    t2 = parse_tag("cp_top02_mg_BASE_seed7_Li4Mg_sg12.cif")
    assert t2.target == "cp"
    assert t2.backbone == "mg"
    assert t2.backbone_name == "MatterGen"
    assert t2.policy == "BASE"
    assert t2.seed == 7
    assert t2.rank == 2
    assert t2.spacegroup == 12


def test_parse_tag_cf_backbone():
    from apu_synthesizability.score_structures import parse_tag
    t = parse_tag("bm_top08_cf_ACC_seed99_Re5W_sg1.cif")
    assert t.backbone == "cf"
    assert t.backbone_name == "CrystalFlow"


def test_parse_tag_formula_with_parens():
    """Formula with parentheses like Mn(BMo)3 must be parsed correctly."""
    from apu_synthesizability.score_structures import parse_tag
    t = parse_tag("bm_top16_adit_BASE_seed23_Mn(BMo)3_sg63.cif")
    assert t.formula == "Mn(BMo)3"
    assert t.spacegroup == 63
    assert t.backbone == "adit"

    t2 = parse_tag("bm_top17_adit_ACC_seed23_FeRe(PW)2_sg26.cif")
    assert t2.formula == "FeRe(PW)2"
    assert t2.spacegroup == 26


def test_parse_tag_formula_complex():
    """Multi-group formula like Li7(MgAl)2."""
    from apu_synthesizability.score_structures import parse_tag
    t = parse_tag("cp_top10_mg_BASE_seed23_Li7(MgAl)2_sg12.cif")
    assert t.formula == "Li7(MgAl)2"
    assert t.spacegroup == 12


# ---------------------------------------------------------------------------
# concordance tests
# ---------------------------------------------------------------------------

def test_concordance_perfect():
    from apu_synthesizability.score_structures import concordance
    c = concordance(np.array([0.1, 0.4, 0.8, 0.9]), np.array([0.2, 0.5, 0.7, 0.95]))
    assert c["spearman"] > 0.99
    assert c["agree_gt_half"] == 1.0
    assert c["n"] == 4


def test_concordance_threshold_agreement():
    from apu_synthesizability.score_structures import concordance
    # Both pairs disagree on >0.5 side: 0.2<0.5 vs 0.6>0.5; 0.8>0.5 vs 0.4<0.5
    c = concordance(np.array([0.2, 0.8]), np.array([0.6, 0.4]))
    assert c["agree_gt_half"] == 0.0
    assert c["n"] == 2


def test_concordance_partial_agreement():
    from apu_synthesizability.score_structures import concordance
    # First pair agrees (both >0.5), second disagrees (0.3<0.5 vs 0.7>0.5)
    c = concordance(np.array([0.8, 0.3]), np.array([0.6, 0.7]))
    assert c["agree_gt_half"] == 0.5
    assert c["n"] == 2


def test_concordance_spearman_negative():
    from apu_synthesizability.score_structures import concordance
    # Perfectly anti-correlated scores
    c = concordance(np.array([0.1, 0.3, 0.7, 0.9]), np.array([0.9, 0.7, 0.3, 0.1]))
    assert c["spearman"] < -0.99


def test_concordance_keys():
    from apu_synthesizability.score_structures import concordance
    c = concordance(np.array([0.5, 0.6]), np.array([0.4, 0.7]))
    assert set(c.keys()) == {"spearman", "agree_gt_half", "n"}


# ---------------------------------------------------------------------------
# Real CIF directory test (skipped if dir absent)
# ---------------------------------------------------------------------------

def test_load_real_cifs():
    d = "/Volumes/SSD1_SMAAA/matinvent-hcap-bo/analysis/top_structures/structures"
    if not os.path.isdir(d):
        pytest.skip("cif dir not present")
    from apu_synthesizability.score_structures import load_structures
    items = load_structures(d)
    assert len(items) >= 20
    assert all(hasattr(t, "backbone_name") for t, _ in items)
    # Verify all backbone_name values are in the known set
    known_names = {"MatterGen", "CrystalFlow", "ADiT"}
    assert all(t.backbone_name in known_names for t, _ in items)
    # Verify both targets present
    targets = {t.target for t, _ in items}
    assert "bm" in targets and "cp" in targets
