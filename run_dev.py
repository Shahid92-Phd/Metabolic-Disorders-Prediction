"""
Stage 1 - development-set analyses (nested CV). The held-out test set is NOT touched here.

  python run_dev.py OB|DB|MS|HT

Outputs (out/<NAME>_*):
  split.npz            train/test indices of the stratified 70:30 split
  dev_candidates.csv   outer-fold metrics of the 12 tuned candidate learners (nested CV)
  dev_params.json      per-outer-fold selected hyperparameters
  dev_stacks.csv       outer-fold metrics of stacking variants (ablation)
  fold_stacks.joblib   fitted outer-fold primary stacks + their outer-test indices (for SHAP stability)
"""
import os
import sys, json, time
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_selection import RFECV
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from common import (load, learners, make_pipe, make_pre, compute_metrics, scores_of,
                    mcc_scorer, SEED)

NAME = sys.argv[1]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", f"{NAME}_")
OUTER_K, INNER_K, STACK_K = 5, 3, 5
X, y, spec = load(NAME)
K = y.nunique()
idx = np.arange(len(y))
tr, te = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
np.savez(OUT + "split.npz", train=tr, test=te)
Xd, yd = X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True)
cols = list(X.columns)
L = learners()
outer = StratifiedKFold(OUTER_K, shuffle=True, random_state=SEED)
folds = list(outer.split(Xd, yd))


def evaluate(model, Xt, yt):
    pred = model.predict(Xt)
    sc = scores_of(model, Xt)
    proba = hasattr(model, "predict_proba") and not (
        hasattr(model, "named_steps") and type(model.named_steps["clf"]).__name__ == "RidgeClassifier")
    return compute_metrics(yt, pred, sc, K, proba=proba)


# ------------------------------------------------ 1. nested CV of the 12 candidates
rows, params = [], {n: [] for n in L}
t0 = time.time()
import os
RESUME = os.path.exists(OUT + "dev_candidates.csv") and os.path.exists(OUT + "dev_params.json")
if RESUME:
    print(f"[{NAME}] resuming: candidate results loaded from disk", flush=True)
for f, (a, b) in enumerate([] if RESUME else folds):
    for n, (est, grid) in L.items():
        gs = GridSearchCV(make_pipe(spec, cols, est), {f"clf__{k}": v for k, v in grid.items()},
                          cv=StratifiedKFold(INNER_K, shuffle=True, random_state=SEED),
                          scoring=mcc_scorer(), n_jobs=2, refit=True)
        gs.fit(Xd.iloc[a], yd.iloc[a])
        m = evaluate(gs.best_estimator_, Xd.iloc[b], yd.iloc[b])
        rows.append({"model": n, "fold": f, **m})
        params[n].append({k.replace("clf__", ""): v for k, v in gs.best_params_.items()})
    print(f"[{NAME}] candidates fold {f} done  {time.time()-t0:.0f}s", flush=True)
if RESUME:
    cand = pd.read_csv(OUT + "dev_candidates.csv"); params = json.load(open(OUT + "dev_params.json"))
else:
    cand = pd.DataFrame(rows)
    cand.to_csv(OUT + "dev_candidates.csv", index=False)
    json.dump(params, open(OUT + "dev_params.json", "w"), indent=1, default=str)
rank = (cand.groupby("model")[["MCC", "ROC-AUC"]].mean()
        .sort_values(["MCC", "ROC-AUC"], ascending=False))
order = list(rank.index)
top6, top3 = order[:6], order[:3]
json.dump({"order": order, "top6": top6}, open(OUT + "selection.json", "w"), indent=1)
print(f"[{NAME}] ranking by nested-CV MCC: {order}", flush=True)


# ------------------------------------------------ 2. stacking variants (ablation)
def stack(names, fold, tuned=True, smote=None, cols_=None):
    cols_ = cols_ or cols
    ests = []
    for n in names:
        est = clone(L[n][0])
        if tuned:
            est.set_params(**params[n][fold])
        ests.append((n, make_pipe(spec, cols_, est, smote=smote)))
    return StackingClassifier(ests, final_estimator=LogisticRegression(max_iter=3000),
                              cv=StratifiedKFold(STACK_K, shuffle=True, random_state=SEED),
                              stack_method="auto", n_jobs=2)


variants = {"Top-6 stack (proposed)": dict(names=top6),
            "Top-3 stack": dict(names=top3),
            "All-12 stack": dict(names=order),
            "Top-6 stack, untuned": dict(names=top6, tuned=False)}
if spec["smote"]:
    variants["Top-6 stack, no SMOTE"] = dict(names=top6, smote=False)
variants["Top-6 stack + in-fold RFE"] = dict(names=top6, rfe=True)

srows, fold_models, rfe_sel = [], [], []
for f, (a, b) in enumerate(folds):
    for v, kw in variants.items():
        kw = dict(kw)
        cols_ = cols
        if kw.pop("rfe", False):
            pre = make_pre(spec, cols).fit(Xd.iloc[a])
            Z = pre.transform(Xd.iloc[a]); names = pre.get_feature_names_out()
            sel = RFECV(RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=2),
                        step=1, min_features_to_select=3, scoring=mcc_scorer(),
                        cv=StratifiedKFold(3, shuffle=True, random_state=SEED)).fit(Z, yd.iloc[a])
            kept = set()
            for nm, s in zip(names, sel.support_):
                if s:
                    kept.add(next(c for c in cols if nm == c or nm.startswith(c + "_")))
            cols_ = [c for c in cols if c in kept]
            rfe_sel.append({"fold": f, "kept": cols_, "dropped": [c for c in cols if c not in kept]})
        model = stack(fold=f, cols_=cols_, **kw).fit(Xd.iloc[a][cols_], yd.iloc[a])
        m = compute_metrics(yd.iloc[b], model.predict(Xd.iloc[b][cols_]),
                            scores_of(model, Xd.iloc[b][cols_]), K)
        srows.append({"model": v, "fold": f, **m})
        if v == "Top-6 stack (proposed)":
            fold_models.append({"model": model, "test_idx": b, "train_idx": a})
    print(f"[{NAME}] stacks fold {f} done  {time.time()-t0:.0f}s", flush=True)
stacks = pd.DataFrame(srows)
stacks.to_csv(OUT + "dev_stacks.csv", index=False)
json.dump(rfe_sel, open(OUT + "dev_rfe.json", "w"), indent=1)
joblib.dump(fold_models, OUT + "fold_stacks.joblib", compress=3)
print(f"[{NAME}] DONE {time.time()-t0:.0f}s")
print(pd.concat([cand, stacks]).groupby("model")[["MCC", "ROC-AUC", "Accuracy"]].mean()
      .sort_values("MCC", ascending=False).round(4))
