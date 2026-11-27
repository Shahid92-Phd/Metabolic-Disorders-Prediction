"""
Stage 3 - explainability of the FULL fitted stacking pipeline (raw features in, probability out).

  python run_xai.py OB|DB|MS|HT

* SHAP: model-agnostic permutation explainer applied to the complete stack (preprocessing,
  base learners and LR meta-learner), background = 50 random development records,
  explained = 150 random test records (stratified), output = P(positive class) for binary
  tasks and the probability of each class for obesity (importance summed over classes).
* Stability: the same explainer applied to the five outer-fold stacks of Stage 1.
* Correlated predictors: grouped permutation importance on the test set (groups with |rho|>=0.6).
* Meta-learner: LR coefficients and mean |coef x centred base score| on the test set.
* LIME: one representative correctly classified case (closest to the median predicted
  probability among true positives), 5000 perturbation samples, 10 features.
"""
import os
import sys, json, time, warnings
import numpy as np, pandas as pd, joblib, shap
from scipy import stats
from sklearn import metrics as Mx
from sklearn.model_selection import train_test_split
from common import load, SEED

warnings.filterwarnings("ignore")
NAME = sys.argv[1]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", f"{NAME}_")
X, y, spec = load(NAME)
K = y.nunique()
sp = np.load(OUT + "split.npz"); tr, te = sp["train"], sp["test"]
Xd, yd = X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True)
Xt, yt = X.iloc[te].reset_index(drop=True), y.iloc[te].reset_index(drop=True)
cols = list(X.columns)
stack = joblib.load(OUT + "final_stack.joblib")
t0 = time.time()

# numeric codes for nominal string columns so that SHAP/LIME can perturb them
codes = {c: sorted(X[c].dropna().unique().tolist()) for c in spec["nominal"]}


def encode(df):
    df = df.copy()
    for c, cats in codes.items():
        df[c] = df[c].map({v: i for i, v in enumerate(cats)})
    return df.astype(float)


def decode(arr):
    df = pd.DataFrame(arr, columns=cols)
    for c, cats in codes.items():
        df[c] = df[c].map(lambda v: cats[int(round(v))] if pd.notna(v) else np.nan)
    for c in cols:
        if c not in codes:
            df[c] = df[c].astype(float)
    return df


def fn(model):
    def f(a):
        p = model.predict_proba(decode(a))
        return p[:, 1] if K == 2 else p
    return f


rng = np.random.default_rng(SEED)
bg = encode(Xd.iloc[rng.choice(len(Xd), 50, replace=False)]).values
_, ex_idx = train_test_split(np.arange(len(Xt)), test_size=min(150, len(Xt) - K), stratify=yt,
                             random_state=SEED)
ex = encode(Xt.iloc[ex_idx]).values


def shap_imp(model, data, evals):
    e = shap.PermutationExplainer(fn(model), shap.maskers.Independent(bg, max_samples=50),
                                  seed=SEED)
    sv = e(data, max_evals=evals, silent=True)
    v = np.abs(sv.values)
    imp = v.mean(0) if v.ndim == 2 else v.sum(2).mean(0)   # multiclass: sum over classes
    return pd.Series(imp, index=cols), sv


M_ = len(cols)
imp, sv = shap_imp(stack, ex, evals=2 * M_ * 6 + 1)
res = {"shap_mean_abs": imp.sort_values(ascending=False).to_dict(),
       "shap_settings": {"explainer": "shap.PermutationExplainer", "shap_version": shap.__version__,
                         "background": "50 random development-set records",
                         "explained": f"{len(ex_idx)} stratified random test-set records",
                         "max_evals": 2 * M_ * 6 + 1,
                         "output": "P(class 1)" if K == 2 else "P(each class); |SHAP| summed over classes",
                         "model": "complete fitted stacking pipeline (raw predictors in)"}}
if K == 2:   # signed summary: correlation of SHAP with standardised feature value (direction)
    vals = ex.copy()
    res["shap_direction"] = {c: float(pd.Series(sv.values[:, j]).corr(pd.Series(vals[:, j]),
                             method="spearman")) for j, c in enumerate(cols) if c not in codes}
np.save(OUT + "shap_values.npy", sv.values); np.save(OUT + "shap_data.npy", ex)
print(f"[{NAME}] SHAP final done {time.time()-t0:.0f}s", flush=True)

# ---------------- stability across the five outer-fold stacks (common 100 test cases)
folds = joblib.load(OUT + "fold_stacks.joblib")
sub = ex[:100]
ranks = []
for fm in folds:
    i_, _ = shap_imp(fm["model"], sub, evals=2 * M_ * 3 + 1)
    ranks.append(i_)
R = pd.DataFrame(ranks).T            # features x folds (importance)
RK = R.rank(ascending=False)
m, n = RK.shape[1], RK.shape[0]
Rsum = RK.sum(1); W = 12 * ((Rsum - Rsum.mean()) ** 2).sum() / (m ** 2 * (n ** 3 - n))
rhos = [stats.spearmanr(R.iloc[:, a], R.iloc[:, b])[0] for a in range(m) for b in range(a + 1, m)]
top5 = {c: int(sum(c in R.iloc[:, j].nlargest(5).index for j in range(m))) for c in cols}
res["stability"] = {"kendall_W": float(W), "mean_pairwise_spearman": float(np.mean(rhos)),
                    "min_pairwise_spearman": float(np.min(rhos)),
                    "top5_frequency_out_of_5": top5,
                    "fold_importance": R.to_dict()}
print(f"[{NAME}] stability W={W:.3f} done {time.time()-t0:.0f}s", flush=True)

# ---------------- correlated predictors: groups and grouped permutation importance
num = [c for c in cols if c not in codes]
rho = Xd[num].corr(method="spearman").abs()
groups, seen = [], set()
for c in num:
    if c in seen:
        continue
    g = [c] + [d for d in num if d != c and d not in seen and rho.loc[c, d] >= 0.6]
    if len(g) > 1:
        groups.append(g); seen.update(g)
res["correlated_pairs"] = [{"a": a, "b": b, "rho": float(rho.loc[a, b])}
                           for i, a in enumerate(num) for b in num[i + 1:] if rho.loc[a, b] >= 0.5]
units = [[c] for c in cols if c not in seen] + groups


def score(model, df):
    p = model.predict_proba(df)
    return Mx.roc_auc_score(yt, p[:, 1]) if K == 2 else Mx.roc_auc_score(yt, p, multi_class="ovr")


base = score(stack, Xt); gp = []
for u in units:
    drops = []
    for r in range(10):
        Z = Xt.copy()
        perm = np.random.default_rng(SEED + r).permutation(len(Z))
        Z[u] = Z[u].values[perm]          # permute the group jointly
        drops.append(base - score(stack, Z))
    gp.append({"features": " + ".join(u), "auc_drop_mean": float(np.mean(drops)),
               "auc_drop_sd": float(np.std(drops, ddof=1))})
res["grouped_permutation"] = sorted(gp, key=lambda d: -d["auc_drop_mean"])
print(f"[{NAME}] grouped permutation done {time.time()-t0:.0f}s", flush=True)

# ---------------- meta-learner contributions
names = [n for n, _ in stack.estimators]
Zt = stack.transform(Xt)                       # base-learner outputs fed to the meta-learner
coef = stack.final_estimator_.coef_
per = Zt.shape[1] // len(names)
contrib = {}
Zc = Zt - Zt.mean(0)
for j, n in enumerate(names):
    blk = slice(j * per, (j + 1) * per)
    contrib[n] = float(np.abs(Zc[:, blk] @ coef[:, blk].T).mean())
res["meta"] = {"base_learners": names, "outputs_per_learner": per,
               "coef": coef.tolist(), "mean_abs_contribution": contrib,
               "base_output_corr": pd.DataFrame(Zt[:, ::per] if K == 2 else Zt[:, ::per],
                                                columns=names).corr().round(3).to_dict()}

# ---------------- LIME (one representative case)
from lime.lime_tabular import LimeTabularExplainer
dev_enc = encode(Xd).fillna(encode(Xd).median())
cat_idx = [cols.index(c) for c in spec["binary"] + spec["nominal"]]
lime = LimeTabularExplainer(dev_enc.values, feature_names=cols, class_names=spec["classes"],
                            categorical_features=cat_idx, discretize_continuous=True,
                            mode="classification", random_state=SEED)
P = stack.predict_proba(Xt); pr = P.argmax(1)
target = 1 if K == 2 else 4       # obesity: Obesity type I
cand = np.where((yt.values == target) & (pr == target))[0]
med = np.median(P[cand, target])
i0 = cand[np.argmin(np.abs(P[cand, target] - med))]
row = encode(Xt.iloc[[i0]]).values[0]
row = np.where(np.isnan(row), dev_enc.median().values, row)


def lime_fn(a):
    return stack.predict_proba(decode(a))


expl = lime.explain_instance(row, lime_fn, labels=(target,), num_features=10, num_samples=5000)
res["lime"] = {"test_position": int(i0), "dataset_row": int(te[i0]), "class_explained": spec["classes"][target],
               "predicted_probability": float(P[i0, target]), "true_class": spec["classes"][int(yt[i0])],
               "weights": expl.as_list(label=target), "local_r2": float(expl.score),
               "settings": {"num_samples": 5000, "num_features": 10, "discretize_continuous": True,
                            "training_data": "development set, median-imputed", "random_state": SEED}}
fig = expl.as_pyplot_figure(label=target)
fig.set_size_inches(7, 4); fig.tight_layout()
fig.savefig(OUT + "lime.png", dpi=300);
json.dump(res, open(OUT + "xai.json", "w"), indent=1, default=float)
print(f"[{NAME}] XAI ALL DONE {time.time()-t0:.0f}s")
