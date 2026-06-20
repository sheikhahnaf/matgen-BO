import numpy as np, pandas as pd
from apu_synthesizability.dataset import assign_splits

def test_assign_splits_are_disjoint_and_stratified():
    df = pd.DataFrame({"label": [1]*100 + [0]*300})
    out = assign_splits(df, fracs=(0.7,0.1,0.2), seed=0)
    assert set(out["split"]) == {"train","val","test"}
    # stratified: test has ~20% of positives
    assert 15 <= (out[(out.split=="test") & (out.label==1)].shape[0]) <= 25
