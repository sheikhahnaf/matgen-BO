import json, pandas as pd
from apu_synthesizability.select import load_results, write_leaderboard, select_best

def _write(d, p):
    p.write_text(json.dumps(d))

def test_select_best_prefers_deployable_high_auprc(tmp_path):
    rdir = tmp_path/"results"; rdir.mkdir()
    _write({"name":"a","deployable":True,"proxy_auprc":0.80,"ece":0.20,"arch":"rf","features":["orb_pca"]}, rdir/"a.json")
    _write({"name":"b","deployable":True,"proxy_auprc":0.80,"ece":0.10,"arch":"xgboost","features":["orb_pca","magpie"]}, rdir/"b.json")
    _write({"name":"c","deployable":False,"proxy_auprc":0.95,"ece":0.05,"arch":"xgboost","features":["orb_pca","mp_props"]}, rdir/"c.json")
    df = load_results(str(rdir))
    assert len(df) == 3
    best = select_best(df)
    assert best["name"] == "b"          # deployable, AUPRC tie broken by lower ECE
    # non-deployable c excluded from the pick despite higher AUPRC
    assert best["deployable"] == True

def test_write_leaderboard_sorted_csv(tmp_path):
    rdir = tmp_path/"results"; rdir.mkdir()
    _write({"name":"a","deployable":True,"proxy_auprc":0.5,"ece":0.2}, rdir/"a.json")
    _write({"name":"b","deployable":True,"proxy_auprc":0.9,"ece":0.1}, rdir/"b.json")
    df = load_results(str(rdir))
    out = tmp_path/"leaderboard.csv"
    write_leaderboard(df, str(out))
    rows = pd.read_csv(out)
    assert list(rows["name"]) == ["b","a"]   # sorted by AUPRC desc
