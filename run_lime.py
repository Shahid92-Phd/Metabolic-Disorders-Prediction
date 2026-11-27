"""Re-run only the LIME part with human-readable category names (updates <NAME>_xai.json and <NAME>_lime.png)."""
import os
import sys, json, numpy as np, pandas as pd, joblib
from lime.lime_tabular import LimeTabularExplainer
from common import load, SEED

NAME = sys.argv[1]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", f"{NAME}_")
X, y, spec = load(NAME); K = y.nunique()
sp = np.load(OUT + "split.npz"); tr, te = sp["train"], sp["test"]
Xd = X.iloc[tr].reset_index(drop=True); Xt = X.iloc[te].reset_index(drop=True); yt = y.iloc[te].reset_index(drop=True)
cols = list(X.columns)
stack = joblib.load(OUT + "final_stack.joblib")
codes = {c: sorted(X[c].dropna().unique().tolist()) for c in spec["nominal"]}
BIN = {"OB": {"FH": ["no", "yes"], "FF": ["no", "yes"]}, "DB": {},
       "MS": {"GD": ["female", "male"]},
       "HT": {"male": ["female", "male"], "currentSmoker": ["no", "yes"], "BPMeds": ["no", "yes"],
              "diabetes": ["no", "yes"]}}[NAME]


def encode(df):
    df = df.copy()
    for c, cats in codes.items():
        df[c] = df[c].map({v: i for i, v in enumerate(cats)})
    return df.astype(float)


def decode(a):
    df = pd.DataFrame(a, columns=cols)
    for c, cats in codes.items():
        df[c] = df[c].map(lambda v: cats[int(round(v))] if pd.notna(v) else np.nan)
    for c in cols:
        if c not in codes:
            df[c] = df[c].astype(float)
    return df


dev = encode(Xd); dev = dev.fillna(dev.median())
cat_idx = [cols.index(c) for c in spec["binary"] + spec["nominal"]]
cat_names = {cols.index(c): [str(v) for v in v_] for c, v_ in {**codes, **BIN}.items()}
lime = LimeTabularExplainer(dev.values, feature_names=cols, class_names=spec["classes"],
                            categorical_features=cat_idx, categorical_names=cat_names,
                            discretize_continuous=True, mode="classification", random_state=SEED)
P = stack.predict_proba(Xt); pr = P.argmax(1)
target = 1 if K == 2 else 4
cand = np.where((yt.values == target) & (pr == target))[0]
i0 = cand[np.argmin(np.abs(P[cand, target] - np.median(P[cand, target])))]
row = encode(Xt.iloc[[i0]]).values[0]
row = np.where(np.isnan(row), dev.median().values, row)
ex = lime.explain_instance(row, lambda a: stack.predict_proba(decode(a)), labels=(target,),
                           num_features=10, num_samples=5000)
res = json.load(open(OUT + "xai.json"))
res["lime"] = {"test_position": int(i0), "dataset_row": int(te[i0]), "class_explained": spec["classes"][target],
               "predicted_probability": float(P[i0, target]), "true_class": spec["classes"][int(yt[i0])],
               "weights": ex.as_list(label=target), "local_r2": float(ex.score),
               "instance": {c: (None if pd.isna(v) else (v if isinstance(v, str) else float(v)))
                            for c, v in Xt.iloc[i0].items()},
               "settings": {"num_samples": 5000, "num_features": 10, "discretize_continuous": True,
                            "training_data": "development set, median-imputed", "random_state": SEED}}
fig = ex.as_pyplot_figure(label=target)
ax = fig.axes[0]
ax.set_title(f"{spec['title']}: P({spec['classes'][target]}) = {P[i0, target]:.2f}", fontsize=10)
fig.set_size_inches(7, 4); fig.tight_layout(); fig.savefig(OUT + "lime.png", dpi=300)
json.dump(res, open(OUT + "xai.json", "w"), indent=1, default=float)
print(NAME, res["lime"]["weights"], res["lime"]["local_r2"])
