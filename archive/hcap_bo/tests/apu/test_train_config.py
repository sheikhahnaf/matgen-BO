"""Tests for apu_synthesizability.train_config — end-to-end config runner."""
import json
import numpy as np
import pytest

from apu_synthesizability.train_config import run_config


def _toy_bank(tmp_path):
    """Separable, CLEAN toy: positives ~ N(+2,1); unlabeled are true negatives
    ~ N(-2,1) only (no hidden positives).  This makes the held-out eval
    (test_pos vs test_unl) unambiguous so AUPRC is genuinely high (~1.0).
    """
    rng = np.random.default_rng(0)
    n_pos, n_unl = 120, 360
    orb_pos = rng.normal(2, 1, size=(n_pos, 8))
    orb_unl = rng.normal(-2, 1, size=(n_unl, 8))   # clean true-negatives only
    orb = np.vstack([orb_pos, orb_unl])
    label = np.concatenate([np.ones(n_pos), np.zeros(n_unl)]).astype(int)
    # stratified-ish split: ~25% test per class
    split = np.array(["train"] * (n_pos + n_unl), dtype=object)
    rngi = np.random.default_rng(1)
    for lab in (0, 1):
        idx = np.where(label == lab)[0]
        rngi.shuffle(idx)
        split[idx[: int(0.25 * len(idx))]] = "test"
    magpie = rng.normal(0, 1, size=(n_pos + n_unl, 12))
    p = tmp_path / "bank.npz"
    np.savez(
        p,
        material_id=np.arange(n_pos + n_unl).astype(str),
        orb_pca=orb,
        magpie=magpie,
        label=label,
        split=split,
    )
    return str(p)


def test_run_config_writes_metrics(tmp_path):
    bank = _toy_bank(tmp_path)
    cfg = {
        "name": "t",
        "features": ["orb_pca"],
        "arch": "rf",
        "pu_scheme": "mv_bagging",
        "n_bags": 8,
        "deployable": True,
        "seed": 0,
    }
    out = tmp_path / "r.json"
    res = run_config(cfg, bank, str(out))
    assert out.exists()
    d = json.load(open(out))
    for k in ("proxy_auprc", "ece", "tpr_on_labeled", "proxy_auroc"):
        assert k in d, f"Missing key '{k}' in result"
    assert 0.0 <= d["proxy_auprc"] <= 1.0
    # separable toy: AUPRC should be well above chance
    assert d["proxy_auprc"] > 0.7, f"proxy_auprc={d['proxy_auprc']:.3f} too low for separable toy"


def test_run_config_returns_dict_with_cfg_keys(tmp_path):
    """Result dict should echo back the config keys."""
    bank = _toy_bank(tmp_path)
    cfg = {
        "name": "test_cfg",
        "features": ["magpie"],
        "arch": "rf",
        "n_bags": 5,
        "seed": 42,
    }
    out = tmp_path / "r2.json"
    res = run_config(cfg, bank, str(out))
    assert res["name"] == "test_cfg"
    assert res["arch"] == "rf"
    assert "n_train" in res
    assert "n_test" in res
    assert res["n_train"] > 0
    assert res["n_test"] > 0


def test_run_config_missing_feature_raises(tmp_path):
    """Requesting a feature block absent from bank should raise ValueError."""
    bank = _toy_bank(tmp_path)
    cfg = {
        "name": "bad",
        "features": ["cgnf_score"],
        "arch": "rf",
        "seed": 0,
    }
    out = tmp_path / "bad.json"
    with pytest.raises((ValueError, KeyError)):
        run_config(cfg, bank, str(out))


def test_run_config_nnpu(tmp_path):
    """nnpu arch should also produce valid metrics."""
    bank = _toy_bank(tmp_path)
    cfg = {
        "name": "nnpu_test",
        "features": ["orb_pca"],
        "arch": "nnpu",
        "seed": 0,
    }
    out = tmp_path / "nnpu.json"
    res = run_config(cfg, bank, str(out))
    assert out.exists()
    assert "proxy_auprc" in res
    assert 0.0 <= res["proxy_auprc"] <= 1.0


def test_run_config_multi_feature(tmp_path):
    """Concatenating multiple feature blocks should work."""
    bank = _toy_bank(tmp_path)
    cfg = {
        "name": "multi",
        "features": ["orb_pca", "magpie"],
        "arch": "rf",
        "n_bags": 5,
        "seed": 0,
    }
    out = tmp_path / "multi.json"
    res = run_config(cfg, bank, str(out))
    assert out.exists()
    assert "proxy_auprc" in res
