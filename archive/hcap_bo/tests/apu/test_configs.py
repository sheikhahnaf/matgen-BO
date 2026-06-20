from configs.apu._gen import build_configs


def test_config_grid():
    cfgs = build_configs()
    assert 20 <= len(cfgs) <= 30
    deployable = [c for c in cfgs if c["deployable"]]
    assert len(deployable) >= 20
    for c in cfgs:
        assert c["features"] and c["arch"] in {"xgboost", "rf", "mlp", "nnpu"}
        assert "name" in c and "pu_scheme" in c


def test_config_names_unique():
    cfgs = build_configs()
    names = [c["name"] for c in cfgs]
    assert len(names) == len(set(names))
