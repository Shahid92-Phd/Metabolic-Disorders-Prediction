"""
Stage 2 - final model, single held-out test evaluation, and sensitivity analyses.

  python run_final.py OB|DB|MS|HT

Uses the learner ranking fixed in Stage 1 (development data only). The test set is used
once per reported configuration and never for selection.
"""
import os
import sys, json, time
import numpy as np, pandas as pd, joblib
from scipy import stats
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn import metrics as Mx
from common import (load, learners, make_pipe, make_pre, compute_metrics, scores_of,
                    mcc_scorer, cal_slope_intercept, SEED, DATA)

NAME = sys.argv[1]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", f"{NAME}_")
X, y, spec = load(NAME)
K = y.nunique()
sp = np.load(OUT + "split.npz"); tr, te = sp["train"], sp["test"]
sel = json.load(open(OUT + "selection.json")); top6 = sel["top6"]
L = learners()
cols = list(X.columns)
t0 = time.time()


def tune(names, cols_, Xd, yd, impute="median", cap=True, smote=None):
    best = {}
    for n in names:
        est, grid = L[n]
        gs = GridSearchCV(make_pipe(spec, cols_, est, smote=smote, impute=impute, cap=cap),
                          {f"clf__{k}": v for k, v in grid.items()},
                          cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                          scoring=mcc_scorer(), n_jobs=2).fit(Xd[cols_], yd)
        best[n] = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
    return best


def build(names, best, cols_, impute="median", cap=True, smote=None):
    ests = [(n, make_pipe(spec, cols_, clone(L[n][0]).set_params(**best[n]),
                          smote=smote, impute=impute, cap=cap)) for n in names]
    return StackingClassifier(ests, final_estimator=LogisticRegression(max_iter=3000),
                              cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                              stack_method="auto", n_jobs=2)


def single(n, best, cols_):
    return make_pipe(spec, cols_, clone(L[n][0]).set_params(**best[n]))


def is_proba(model):
    return not (hasattr(model, "named_steps") and
                type(model.named_steps["clf"]).__name__ == "RidgeClassifier")


def boot_ci(yt, pred, sc, proba=True, B=2000):
    rng = np.random.default_rng(SEED)
    n = len(yt); res = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if len(np.unique(yt[i])) < K:
            continue
        m = compute_metrics(yt[i], pred[i], sc[i], K, proba=proba)
        m.pop("ECE", None)
        res.append(m)
    df = pd.DataFrame(res)
    return df.quantile(0.025), df.quantile(0.975)


# ----- DeLong (Sun & Xu fast implementation) for paired binary AUCs
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N); T2[J] = T
    return T2


def delong(y, s1, s2):
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()
    aucs, V10, V01 = [], [], []
    for s in (s1, s2):
        tx, ty, tz = _midrank(s[pos]), _midrank(s[neg]), _midrank(s)
        aucs.append(tz[pos].sum() / m / n - (m + 1.0) / 2.0 / n)
        V10.append((tz[pos] - tx) / n)
        V01.append(1.0 - (tz[neg] - ty) / m)
    S = np.cov(np.vstack(V10)) / m + np.cov(np.vstack(V01)) / n
    d = aucs[0] - aucs[1]
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = d / np.sqrt(var) if var > 0 else 0.0
    return aucs[0], aucs[1], 2 * stats.norm.sf(abs(z))


def mcnemar(yt, p1, p2):
    a = (p1 == yt) & (p2 != yt); b = (p1 != yt) & (p2 == yt)
    n01, n10 = a.sum(), b.sum()
    if n01 + n10 == 0:
        return 1.0
    return stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue


Xd, yd = X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True)
Xt, yt = X.iloc[te].reset_index(drop=True), y.iloc[te].reset_index(drop=True).values
report = {"dataset": NAME, "top6": top6, "n_dev": len(tr), "n_test": len(te)}

# ------------------------------------------------ class counts at each stage (R3.6)
cc = {"Full dataset": y.value_counts().sort_index().tolist(),
      "Development (70%)": yd.value_counts().sort_index().tolist(),
      "Test (30%)": pd.Series(yt).value_counts().sort_index().tolist()}
if spec["smote"]:
    from imblearn.over_sampling import SMOTE
    Z = make_pre(spec, cols).fit_transform(Xd)
    _, ys = SMOTE(random_state=SEED, k_neighbors=5).fit_resample(Z, yd)
    cc["Development after SMOTE (training only)"] = pd.Series(ys).value_counts().sort_index().tolist()
report["class_counts"] = cc
report["missing_counts"] = X.isna().sum()[X.isna().sum() > 0].to_dict()

# ------------------------------------------------ primary model
best = tune(top6, cols, Xd, yd)
report["final_params"] = best
stack = build(top6, best, cols).fit(Xd, yd)
pred = stack.predict(Xt); sc_full = stack.predict_proba(Xt)
sc = sc_full[:, 1] if K == 2 else sc_full
m = compute_metrics(yt, pred, sc, K)
lo, hi = boot_ci(yt, pred, sc)
report["test_stack"] = m
report["test_stack_ci"] = {"lo": lo.to_dict(), "hi": hi.to_dict()}
report["confusion"] = Mx.confusion_matrix(yt, pred).tolist()
if K == 2:
    s, i = cal_slope_intercept(yt, sc)
    report["calibration"] = {"slope": s, "intercept": i}
    pt, pp = __import__("sklearn.calibration", fromlist=["calibration_curve"]).calibration_curve(
        yt, sc, n_bins=10, strategy="quantile")
    report["cal_curve"] = {"obs": pt.tolist(), "pred": pp.tolist()}
    th = np.round(np.arange(0.01, 0.81, 0.01), 2); n = len(yt); prev = yt.mean()
    nb = []
    for t in th:
        p_ = sc >= t; tp = (p_ & (yt == 1)).sum(); fp = (p_ & (yt == 0)).sum()
        nb.append({"threshold": t, "model": tp / n - fp / n * t / (1 - t),
                   "treat_all": prev - (1 - prev) * t / (1 - t), "treat_none": 0.0})
    report["dca"] = nb
joblib.dump(stack, OUT + "final_stack.joblib", compress=3)
pdf = pd.DataFrame({"test_row_index_in_dataset": te, "y_true": yt, "y_pred": pred})
for k in range(K):
    pdf[f"prob_class_{k}"] = sc_full[:, k]
pdf.to_csv(OUT + "test_predictions.csv", index=False)
meta = stack.final_estimator_
report["meta_coef"] = meta.coef_.tolist(); report["meta_intercept"] = meta.intercept_.tolist()
print(f"[{NAME}] primary done {time.time()-t0:.0f}s  MCC={m['MCC']:.4f} AUC={m['ROC-AUC']:.4f}", flush=True)

# ------------------------------------------------ individual learners on test (comparison)
singles = {}
for n in top6:
    mdl = single(n, best, cols).fit(Xd, yd)
    p1 = mdl.predict(Xt); s1 = scores_of(mdl, Xt)
    mm = compute_metrics(yt, p1, s1, K, proba=is_proba(mdl))
    l1, h1 = boot_ci(yt, p1, s1, proba=is_proba(mdl), B=1000)
    rec = {"metrics": mm, "lo": l1.to_dict(), "hi": h1.to_dict(), "mcnemar_p": mcnemar(yt, pred, p1)}
    if K == 2:
        _, _, rec["delong_p"] = delong(yt, sc, np.asarray(s1, float))
    # paired bootstrap of MCC / AUC difference
    rng = np.random.default_rng(SEED); dm, da = [], []
    for _ in range(1000):
        i = rng.integers(0, len(yt), len(yt))
        if len(np.unique(yt[i])) < K:
            continue
        dm.append(Mx.matthews_corrcoef(yt[i], pred[i]) - Mx.matthews_corrcoef(yt[i], p1[i]))
        if K == 2:
            da.append(Mx.roc_auc_score(yt[i], sc[i]) - Mx.roc_auc_score(yt[i], s1[i]))
    rec["dMCC_ci"] = [float(np.quantile(dm, .025)), float(np.quantile(dm, .975))]
    if K == 2:
        rec["dAUC_ci"] = [float(np.quantile(da, .025)), float(np.quantile(da, .975))]
    singles[n] = rec
report["test_singles"] = singles
print(f"[{NAME}] singles done {time.time()-t0:.0f}s", flush=True)


# ------------------------------------------------ sensitivity analyses (reported, not used for selection)
def sens(label, cols_=None, impute="median", cap=True, Xd_=None, Xt_=None):
    cols_ = cols_ or cols
    Xd_ = Xd if Xd_ is None else Xd_
    Xt_ = Xt if Xt_ is None else Xt_
    b = tune(top6, cols_, Xd_, yd, impute=impute, cap=cap)
    st = build(top6, b, cols_, impute=impute, cap=cap).fit(Xd_[cols_], yd)
    p = st.predict(Xt_[cols_]); s = st.predict_proba(Xt_[cols_]); s = s[:, 1] if K == 2 else s
    mm = compute_metrics(yt, p, s, K); l_, h_ = boot_ci(yt, p, s, B=1000)
    print(f"[{NAME}] sens '{label}' MCC={mm['MCC']:.4f} AUC={mm['ROC-AUC']:.4f} {time.time()-t0:.0f}s", flush=True)
    return {"label": label, "cols": cols_, "metrics": mm, "lo": l_.to_dict(), "hi": h_.to_dict()}


S = []
if spec["label_feats"]:
    S.append(sens("Excluding label-defining predictors: " + ", ".join(spec["label_feats"]),
                  cols_=[c for c in cols if c not in spec["label_feats"]]))
if X.isna().any().any():
    for imp in ["mean", "knn", "iterative"]:
        S.append(sens(f"Imputation: {imp}", impute=imp))
if spec["cap"]:
    S.append(sens("No outlier capping", cap=False))
if NAME == "DB":
    # Reproduce the original data preparation: zeros replaced by class-specific medians on all records
    leak = pd.read_csv(DATA + "Diabetes_Dataset.xls").drop(columns="DB")
    S.append(sens("Original file with class-conditional median imputation (leaky)",
                  Xd_=leak.iloc[tr].reset_index(drop=True), Xt_=leak.iloc[te].reset_index(drop=True)))
report["sensitivity"] = S
json.dump(report, open(OUT + "final.json", "w"), indent=1, default=float)
print(f"[{NAME}] ALL DONE {time.time()-t0:.0f}s")
