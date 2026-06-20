import re, json
import numpy as np, pandas as pd

def read_mp_api_key(notebook_path):
    """Extract the MP API key literal from the A-PU notebook.

    Handles two assignment styles found in practice:
      1. Variable assignment:  api_key = "..."
      2. Dict literal:         "mp_api_key": "..."
    """
    with open(notebook_path) as f:
        nb = json.load(f)
    # Match either  api_key = "VALUE"  or  "mp_api_key": "VALUE"
    pat = re.compile(
        r'(?:api_key\s*=\s*|["\']mp_api_key["\']\s*:\s*)["\']([A-Za-z0-9]{16,})["\']'
    )
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        m = pat.search("".join(c.get("source", [])))
        if m:
            return m.group(1)
    raise RuntimeError("MP API key not found in notebook")

def query_mp(api_key, e_hull_max=0.5, max_pos=60000, max_unl=120000):
    from mp_api.client import MPRester
    rows = []
    with MPRester(api_key) as mpr:
        pos = mpr.materials.summary.search(theoretical=False,
              fields=["material_id","composition","database_IDs"])
        n_pos = 0
        for d in pos:
            ids = d.database_IDs or []
            if any("icsd" in str(x).lower() for x in ids):
                rows.append((str(d.material_id), d.composition.reduced_formula, 1))
                n_pos += 1
            if n_pos >= max_pos:
                break
        unl = mpr.materials.summary.search(theoretical=True,
              energy_above_hull=(0, e_hull_max),
              fields=["material_id","composition","database_IDs"])
        n_unl = 0
        for d in unl:
            ids = d.database_IDs or []
            if any("icsd" in str(x).lower() for x in ids):
                continue
            rows.append((str(d.material_id), d.composition.reduced_formula, 0))
            n_unl += 1
            if n_unl >= max_unl:
                break
    return pd.DataFrame(rows, columns=["material_id","formula","label"]).drop_duplicates("material_id")

def assign_splits(df, fracs=(0.7,0.1,0.2), seed=0):
    assert abs(sum(fracs) - 1.0) < 1e-9, "fracs must sum to 1.0"
    rng = np.random.default_rng(seed); df = df.copy(); df["split"] = "train"
    for lab in df["label"].unique():
        idx = df.index[df["label"] == lab].to_numpy(); rng.shuffle(idx)
        n = len(idx); a = int(fracs[0]*n); b = a + int(fracs[1]*n)
        df.loc[idx[a:b], "split"] = "val"; df.loc[idx[b:], "split"] = "test"
    return df
