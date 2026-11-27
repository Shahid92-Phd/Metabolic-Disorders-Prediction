"""
Stage 4 - assemble every table and figure of the revised manuscript from the saved outputs.
  python make_outputs.py
Writes out/Revision_Tables.xlsx, out/tables.json (for the manuscript builder) and out/fig_*.png
"""
import json, numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn import metrics as Mx
from common import load, learners
import sklearn, imblearn, shap, lime

import os
O = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "")
DS = ["OB", "DB", "MS", "HT"]
TITLE = {"OB": "Obesity", "DB": "Diabetes", "MS": "Metabolic syndrome", "HT": "Hypertension"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})
C1, C2, C3 = "#2f6fae", "#d1495b", "#7a7a7a"
T = {}           # all tables for the manuscript (list-of-rows)
fin = {d: json.load(open(O + f"{d}_final.json")) for d in DS}
xai = {d: json.load(open(O + f"{d}_xai.json")) for d in DS}
cand = {d: pd.read_csv(O + f"{d}_dev_candidates.csv") for d in DS}
stk = {d: pd.read_csv(O + f"{d}_dev_stacks.csv") for d in DS}
sel = {d: json.load(open(O + f"{d}_selection.json")) for d in DS}


def f3(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def pm(a, b):
    return "—" if (a is None or np.isnan(a)) else f"{a:.3f} ± {b:.3f}"


def ci(m, lo, hi, k):
    v = m.get(k, np.nan)
    if v is None or np.isnan(v):
        return "—"
    return f"{v:.3f} ({lo[k]:.3f}–{hi[k]:.3f})"


# ---------------------------------------------------------------- dataset / class counts
rows = []
for d in DS:
    X, y, spec = load(d); f = fin[d]
    cc = f["class_counts"]
    for stage, v in cc.items():
        rows.append([TITLE[d], stage] + [", ".join(f"{spec['classes'][i]}: {n}" for i, n in enumerate(v))])
T["class_counts"] = [["Dataset", "Stage", "Class counts"]] + rows

rows = []
for d in DS:
    X, y, spec = load(d)
    miss = fin[d]["missing_counts"]
    rows.append([TITLE[d], str(len(y)), str(X.shape[1]),
                 "; ".join(f"{k} {v}" for k, v in miss.items()) or "none",
                 f"{fin[d]['n_dev']} / {fin[d]['n_test']}"])
T["datasets"] = [["Dataset", "Records", "Predictors analysed", "Missing values (records)", "Development / test"]] + rows

# ---------------------------------------------------------------- selection (Table 2) & candidates (A6)
T["selection"] = [["Condition", "1st", "2nd", "3rd", "4th", "5th", "6th"]] + \
    [[TITLE[d]] + sel[d]["top6"] for d in DS]
KEYS = ["Accuracy", "Balanced accuracy", "F1", "MCC", "Kappa", "ROC-AUC"]
rows = []
for d in DS:
    g = cand[d].groupby("model")
    mu, sd = g[KEYS].mean(), g[KEYS].std(ddof=1)
    for n in sel[d]["order"]:
        rows.append([TITLE[d], n] + [pm(mu.loc[n,k], sd.loc[n,k]) for k in KEYS])
T["candidates"] = [["Dataset", "Learner"] + KEYS] + rows

# ---------------------------------------------------------------- ablation (dev outer folds)
rows = []
for d in DS:
    g = stk[d].groupby("model", sort=False)
    mu, sd = g[KEYS].mean(), g[KEYS].std(ddof=1)
    best = sel[d]["top6"][0]
    cm, cs = cand[d][cand[d].model == best][KEYS].mean(), cand[d][cand[d].model == best][KEYS].std(ddof=1)
    for n in mu.index:
        rows.append([TITLE[d], n] + [pm(mu.loc[n,k], sd.loc[n,k]) for k in KEYS])
    rows.append([TITLE[d], f"Best single learner ({best})"] + [pm(cm[k], cs[k]) for k in KEYS])
T["ablation"] = [["Dataset", "Configuration"] + KEYS] + rows

# ---------------------------------------------------------------- statistics on outer folds
# Friedman per dataset (blocks = 5 outer folds) on MCC; corrected resampled t-test (Nadeau & Bengio)
k_folds = 5
rows, glob = [], []
for d in DS:
    a = cand[d].pivot(index="fold", columns="model", values="MCC")
    a["Stack"] = stk[d][stk[d].model == "Top-6 stack (proposed)"].set_index("fold")["MCC"]
    fr = stats.friedmanchisquare(*[a[c] for c in a.columns])
    n_tr = fin[d]["n_dev"] * (k_folds - 1) / k_folds; n_te = fin[d]["n_dev"] / k_folds
    res = []
    for c in a.columns:
        if c == "Stack":
            continue
        dlt = a["Stack"] - a[c]
        v = dlt.var(ddof=1)
        t = dlt.mean() / np.sqrt((1 / k_folds + n_te / n_tr) * v) if v > 0 else 0
        p = 2 * stats.t.sf(abs(t), k_folds - 1)
        res.append([c, dlt.mean(), p])
    ps = np.array([r[2] for r in res]); order = np.argsort(ps); m = len(ps)
    adj = np.empty(m); run = 0
    for rank, i in enumerate(order):
        run = max(run, min(1, (m - rank) * ps[i])); adj[i] = run
    for r, pa in zip(res, adj):
        rows.append([TITLE[d], f"Stack vs {r[0]}", f"{r[1]:+.3f}", f"{r[2]:.3f}", f"{pa:.3f}",
                     "Yes" if pa < 0.05 else "No"])
    glob.append([TITLE[d], f"{fr.statistic:.2f}", f"{fr.pvalue:.4f}",
                 ", ".join(f"{c} {v:.2f}" for c, v in a.rank(axis=1, ascending=False).mean()
                           .sort_values().head(5).items())])
T["stats_pairwise"] = [["Dataset", "Comparison", "Mean ΔMCC (stack − learner)", "p (corrected t)",
                        "Holm-adjusted p", "Significant"]] + rows
T["stats_friedman"] = [["Dataset", "Friedman χ² (13 models, 5 folds)", "p", "Best mean ranks (1 = best)"]] + glob
# across datasets (N = 4)
mm = pd.DataFrame({d: pd.concat([cand[d].groupby("model")["MCC"].mean(),
                                 pd.Series({"Stack": stk[d][stk[d].model == "Top-6 stack (proposed)"]["MCC"].mean()})])
                   for d in DS}).T
fr4 = stats.friedmanchisquare(*[mm[c] for c in mm.columns])
T["stats_global"] = {"chi2": float(fr4.statistic), "p": float(fr4.pvalue),
                     "mean_rank": mm.rank(axis=1, ascending=False).mean().sort_values().round(2).to_dict()}

# ---------------------------------------------------------------- main test-set results
MK = ["Accuracy", "Balanced accuracy", "Recall (sensitivity)", "Specificity", "Precision (PPV)", "NPV",
      "F1", "MCC", "Kappa", "ROC-AUC", "PR-AUC", "Brier"]
rows = []
for k in MK:
    r = [k]
    for d in DS:
        f = fin[d]
        r.append(ci(f["test_stack"], f["test_stack_ci"]["lo"], f["test_stack_ci"]["hi"], k)
                 if k in f["test_stack"] else "—")
    rows.append(r)
cal = ["Calibration slope / intercept"] + [
    f"{fin[d]['calibration']['slope']:.2f} / {fin[d]['calibration']['intercept']:.2f}" if "calibration" in fin[d] else "—"
    for d in DS]
ece = ["Expected calibration error"] + [f3(fin[d]["test_stack"].get("ECE", np.nan)) for d in DS]
qwk = ["Quadratic-weighted kappa"] + [f3(fin[d]["test_stack"].get("Quadratic-weighted kappa", np.nan)) for d in DS]
T["test_main"] = [["Metric"] + [TITLE[d] for d in DS]] + rows + [cal, ece, qwk]

rows = []
for d in DS:
    f = fin[d]
    rows.append([TITLE[d], "Top-6 stack", f3(f["test_stack"]["MCC"]), f3(f["test_stack"]["ROC-AUC"]),
                 f3(f["test_stack"].get("Brier", np.nan)), "—", "—", "—"])
    for n, r in f["test_singles"].items():
        mt = r["metrics"]
        rows.append([TITLE[d], n, f3(mt["MCC"]), f3(mt.get("ROC-AUC", np.nan)), f3(mt.get("Brier", np.nan)),
                     f"{r['dMCC_ci'][0]:+.3f} to {r['dMCC_ci'][1]:+.3f}",
                     f"{r['delong_p']:.3f}" if "delong_p" in r else "—", f"{r['mcnemar_p']:.3f}"])
T["test_vs_singles"] = [["Dataset", "Model", "MCC", "ROC-AUC", "Brier", "ΔMCC stack − model (95% CI)",
                         "DeLong p (AUC)", "McNemar p"]] + rows

rows = []
for d in DS:
    f = fin[d]; m0 = f["test_stack"]
    rows.append([TITLE[d], "Primary analysis", ", ".join([c for c in load(d)[0].columns]) and f"{len(load(d)[0].columns)} predictors",
                 ci(m0, f["test_stack_ci"]["lo"], f["test_stack_ci"]["hi"], "MCC"),
                 ci(m0, f["test_stack_ci"]["lo"], f["test_stack_ci"]["hi"], "ROC-AUC"),
                 ci(m0, f["test_stack_ci"]["lo"], f["test_stack_ci"]["hi"], "Accuracy")])
    for s in f["sensitivity"]:
        lab = (s["label"].replace("Imputation: knn", "Imputation: k-nearest neighbours (k = 5)")
               .replace("Imputation: iterative", "Imputation: iterative (chained equations)")
               .replace("Excluding label-defining predictors: GL", "Excluding label-related predictor: GL (2-h glucose)"))
        rows.append([TITLE[d], lab, f"{len(s['cols'])} predictors",
                     ci(s["metrics"], s["lo"], s["hi"], "MCC"), ci(s["metrics"], s["lo"], s["hi"], "ROC-AUC"),
                     ci(s["metrics"], s["lo"], s["hi"], "Accuracy")])
T["sensitivity"] = [["Dataset", "Analysis", "Predictors", "MCC (95% CI)", "ROC-AUC (95% CI)", "Accuracy (95% CI)"]] + rows

# ---------------------------------------------------------------- confusion matrices
X, y, spec = load("OB")
cmo = np.array(fin["OB"]["confusion"])
T["cm_ob"] = [["True \\ predicted"] + spec["classes"]] + [[spec["classes"][i]] + [str(v) for v in cmo[i]] for i in range(7)]
rows = []
for d in ["DB", "MS", "HT"]:
    (tn, fp), (fn, tp) = fin[d]["confusion"]
    rows.append([TITLE[d], str(tp), str(fp), str(fn), str(tn), str(tp + fn), str(tn + fp)])
T["cm_bin"] = [["Dataset", "TP", "FP", "FN", "TN", "Positives", "Negatives"]] + rows

# ---------------------------------------------------------------- hyperparameters (A7)
L = learners(); rows = []
for d in DS:
    for n in sel[d]["top6"]:
        est, grid = L[n]
        fixed = {k: v for k, v in est.get_params().items()
                 if k in ("n_estimators", "solver", "probability", "max_iter", "random_state") and k not in grid}
        rows.append([TITLE[d], f"{n} ({type(est).__name__})",
                     "; ".join(f"{k} ∈ {v}" for k, v in grid.items()),
                     "; ".join(f"{k} = {v}" for k, v in fin[d]["final_params"][n].items()),
                     "; ".join(f"{k} = {v}" for k, v in fixed.items())])
T["hyper"] = [["Dataset", "Learner (scikit-learn class)", "Search space", "Selected (inner 5-fold CV on development set)",
               "Fixed settings"]] + rows
T["versions"] = {"scikit-learn": sklearn.__version__, "imbalanced-learn": imblearn.__version__,
                 "shap": shap.__version__, "lime": "0.2.0.1", "python": "3.13", "numpy": np.__version__,
                 "pandas": pd.__version__}

# ---------------------------------------------------------------- XAI tables
NICE = {"OB": {"AG": "Age", "HT": "Height", "WT": "Weight", "FH": "Family history of overweight",
               "FF": "High-calorie food", "FV": "Vegetable consumption", "NM": "Main meals/day",
               "CM": "Snacking", "CW": "Water intake", "FA": "Physical activity", "TD": "Screen time",
               "CA": "Alcohol", "TU": "Transport mode"},
        "DB": {"PR": "Pregnancies", "GL": "Glucose (2-h OGTT)", "BP": "Diastolic BP", "ST": "Triceps skinfold",
               "IN": "2-h serum insulin", "BMI": "BMI", "DPF": "Diabetes pedigree function", "AG": "Age"},
        "MS": {"AG": "Age", "GD": "Sex", "MT": "Marital status", "IL": "Income", "EB": "Race/ethnicity",
               "WCM": "Waist circumference", "BMI": "BMI", "UC": "Urinary albumin/creatinine",
               "UA": "Uric acid", "BGL": "Fasting glucose", "HDL": "HDL cholesterol", "TL": "Triglycerides"},
        "HT": {"male": "Sex (male)", "age": "Age", "currentSmoker": "Current smoker",
               "cigsPerDay": "Cigarettes/day", "BPMeds": "BP medication", "diabetes": "Diabetes",
               "totChol": "Total cholesterol", "sysBP": "Systolic BP", "diaBP": "Diastolic BP",
               "BMI": "BMI", "heartRate": "Heart rate", "glucose": "Glucose"}}
DOMAIN = {"AG": "Demographic", "age": "Demographic", "GD": "Demographic", "male": "Demographic",
          "MT": "Socioeconomic", "IL": "Socioeconomic", "EB": "Demographic", "PR": "Reproductive",
          "HT": "Anthropometric", "WT": "Anthropometric", "BMI": "Anthropometric", "WCM": "Anthropometric",
          "ST": "Anthropometric", "GL": "Glycemic", "IN": "Glycemic", "BGL": "Glycemic", "glucose": "Glycemic",
          "HDL": "Lipid", "TL": "Lipid", "totChol": "Lipid", "sysBP": "Haemodynamic", "diaBP": "Haemodynamic",
          "BP": "Haemodynamic", "heartRate": "Haemodynamic", "BPMeds": "Treatment/comorbidity",
          "diabetes": "Treatment/comorbidity", "UC": "Renal/urate", "UA": "Renal/urate",
          "FH": "Family history", "DPF": "Family history", "FF": "Diet", "FV": "Diet", "NM": "Diet",
          "CM": "Diet", "CW": "Diet", "CA": "Diet", "FA": "Physical activity/behaviour",
          "TD": "Physical activity/behaviour", "TU": "Physical activity/behaviour",
          "currentSmoker": "Smoking", "cigsPerDay": "Smoking"}
rows = []
for d in DS:
    s = pd.Series(xai[d]["shap_mean_abs"]); tot = s.sum()
    st = xai[d]["stability"]["top5_frequency_out_of_5"]
    for i, (k, v) in enumerate(s.head(8).items()):
        dirn = xai[d].get("shap_direction", {}).get(k)
        rows.append([TITLE[d], str(i + 1), NICE[d][k], DOMAIN[k], f"{v:.4f}", f"{100*v/tot:.1f}",
                     f"{st[k]}/5", "—" if dirn is None or np.isnan(dirn) else f"{dirn:+.2f}"])
T["shap"] = [["Dataset", "Rank", "Predictor", "Domain", "Mean |SHAP|", "Share (%)", "Top-5 in fold models",
              "Direction ρ(SHAP, value)"]] + rows
T["stability"] = [["Dataset", "Kendall's W (5 fold models)", "Mean pairwise Spearman ρ", "Minimum pairwise ρ"]] + \
    [[TITLE[d], f"{xai[d]['stability']['kendall_W']:.3f}", f"{xai[d]['stability']['mean_pairwise_spearman']:.3f}",
      f"{xai[d]['stability']['min_pairwise_spearman']:.3f}"] for d in DS]
rows = []
for d in DS:
    for g in xai[d]["grouped_permutation"][:6]:
        nm = " + ".join(NICE[d].get(p.strip(), p.strip()) for p in g["features"].split("+"))
        rows.append([TITLE[d], nm, f"{g['auc_drop_mean']:.4f} ± {g['auc_drop_sd']:.4f}"])
T["grouped"] = [["Dataset", "Predictor or correlated group (|ρ| ≥ 0.6)", "Test ROC-AUC drop when permuted (mean ± SD, 10 repeats)"]] + rows
T["corr_pairs"] = {d: xai[d]["correlated_pairs"] for d in DS}
rows = []
for d in DS:
    mc = xai[d]["meta"]["mean_abs_contribution"]; tot = sum(mc.values())
    co = np.array(xai[d]["meta"]["coef"])
    for j, (n, v) in enumerate(sorted(mc.items(), key=lambda t: -t[1])):
        idx = xai[d]["meta"]["base_learners"].index(n)
        coef = f"{co[0, idx]:+.2f}" if co.shape[0] == 1 else "multinomial"
        rows.append([TITLE[d], n, coef, f"{v:.3f}", f"{100*v/tot:.1f}"])
T["meta"] = [["Dataset", "Base learner", "Meta-learner coefficient", "Mean |contribution| (log-odds)", "Share (%)"]] + rows

# domain recurrence: domain represented if >=1 predictor in the SHAP top five of the final model;
# robustness: predictor in top five of >=4/5 fold models
doms = ["Anthropometric", "Glycemic", "Lipid", "Haemodynamic", "Demographic", "Diet",
        "Physical activity/behaviour", "Smoking", "Family history", "Renal/urate", "Socioeconomic",
        "Reproductive", "Treatment/comorbidity"]
rows = []
for dm in doms:
    r = [dm]; cnt = 0; cnt_r = 0
    for d in DS:
        s = pd.Series(xai[d]["shap_mean_abs"]); top = list(s.head(5).index)
        st = xai[d]["stability"]["top5_frequency_out_of_5"]
        avail = any(DOMAIN[c] == dm for c in s.index)
        hit = [c for c in top if DOMAIN[c] == dm]
        rob = [c for c in s.index if DOMAIN[c] == dm and st[c] >= 4]
        if hit:
            cnt += 1
        if rob:
            cnt_r += 1
        r.append(("✓ " + ", ".join(NICE[d][c] for c in hit)) if hit else ("–" if avail else "n/a"))
    r += [f"{cnt}/4", f"{cnt_r}/4"]
    if cnt or any(x != "n/a" for x in r[1:5]):
        rows.append(r)
T["domains"] = [["Domain"] + [TITLE[d] for d in DS] + ["Final-model recurrence", "Robust (≥4/5 folds)"]] + rows
T["lime"] = {d: xai[d]["lime"] for d in DS}
T["shap_settings"] = xai["DB"]["shap_settings"]

# ---------------------------------------------------------------- figures
# class distribution
fig, axes = plt.subplots(1, 4, figsize=(12, 3.0))
for ax, d in zip(axes, DS):
    X, y, spec = load(d); cc = fin[d]["class_counts"]
    st = list(cc.keys()); K = len(cc[st[0]]); w = 0.8 / len(st)
    cols_ = [C3, C1, C2, "#e9a03b"]
    for j, s in enumerate(st):
        ax.bar(np.arange(K) + j * w, cc[s], w, label=s, color=cols_[j])
    ax.set_xticks(np.arange(K) + w * (len(st) - 1) / 2)
    ax.set_xticklabels(spec["classes"], rotation=40 if K > 2 else 0, ha="right" if K > 2 else "center", fontsize=7)
    ax.set_title(TITLE[d], fontsize=9)
axes[0].set_ylabel("Records")
h, l = axes[1].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(O + "fig_classes.png", dpi=300, bbox_inches="tight"); plt.close()

# ROC / PR / calibration / DCA
preds = {d: pd.read_csv(O + f"{d}_test_predictions.csv") for d in DS}
fig, axes = plt.subplots(1, 4, figsize=(12, 3.1))
for ax, d in zip(axes, DS):
    p = preds[d]; yt = p.y_true.values
    if d == "OB":
        P = p[[c for c in p.columns if c.startswith("prob_")]].values
        X, y, spec = load(d)
        for k in range(7):
            fpr, tpr, _ = Mx.roc_curve(yt == k, P[:, k]); ax.plot(fpr, tpr, lw=0.9, label=spec["classes"][k])
        ax.legend(fontsize=5.5, frameon=False, loc="lower right")
        auc = fin[d]["test_stack"]["ROC-AUC"]
    else:
        fpr, tpr, _ = Mx.roc_curve(yt, p.prob_class_1); ax.plot(fpr, tpr, color=C1, lw=1.4)
        auc = fin[d]["test_stack"]["ROC-AUC"]
    lo, hi = fin[d]["test_stack_ci"]["lo"]["ROC-AUC"], fin[d]["test_stack_ci"]["hi"]["ROC-AUC"]
    ax.plot([0, 1], [0, 1], ls=":", color=C3, lw=0.8)
    ax.set_title(f"{TITLE[d]}\nAUC {auc:.3f} (95% CI {lo:.3f}–{hi:.3f})", fontsize=8.5)
    ax.set_xlabel("1 − specificity"); ax.set_aspect("equal")
axes[0].set_ylabel("Sensitivity")
fig.tight_layout(); fig.savefig(O + "fig_roc.png", dpi=300, bbox_inches="tight"); plt.close()

fig, axes = plt.subplots(2, 3, figsize=(10, 6.2))
for j, d in enumerate(["DB", "MS", "HT"]):
    p = preds[d]; yt = p.y_true.values; s = p.prob_class_1.values
    cc = fin[d]["cal_curve"]; ax = axes[0, j]
    ax.plot([0, 1], [0, 1], ls=":", color=C3, lw=0.8)
    ax.plot(cc["pred"], cc["obs"], "o-", color=C1, ms=3, lw=1.2)
    ax.set_title(f"{TITLE[d]}: slope {fin[d]['calibration']['slope']:.2f}, intercept {fin[d]['calibration']['intercept']:.2f}\n"
                 f"Brier {fin[d]['test_stack']['Brier']:.3f}", fontsize=8.5)
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed proportion") if j == 0 else None
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    dc = pd.DataFrame(fin[d]["dca"]); ax = axes[1, j]
    ax.plot(dc.threshold, dc.model, color=C1, lw=1.4, label="Stacking model")
    ax.plot(dc.threshold, dc.treat_all, color=C2, lw=1, ls="--", label="Treat all")
    ax.plot(dc.threshold, dc.treat_none, color=C3, lw=1, ls=":", label="Treat none")
    ax.set_ylim(-0.05, max(dc.model.max(), dc.treat_all.max()) + 0.05)
    ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit") if j == 0 else None
    if j == 0:
        ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(O + "fig_calibration_dca.png", dpi=300, bbox_inches="tight"); plt.close()

# SHAP importance with fold variability
fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
for ax, d in zip(axes.ravel(), DS):
    s = pd.Series(xai[d]["shap_mean_abs"]).head(10)[::-1]
    fi = pd.DataFrame(xai[d]["stability"]["fold_importance"])
    ax.barh([NICE[d][k] for k in s.index], s.values, color=C1, height=0.6, label="Final model")
    for i, k in enumerate(s.index):
        ax.plot(fi.loc[k].values, [i] * fi.shape[1], "o", ms=2.5, color=C2, alpha=0.8,
                label="Outer-fold models" if i == 0 else None)
    ax.set_title(f"{TITLE[d]} (Kendall's W = {xai[d]['stability']['kendall_W']:.2f})", fontsize=9)
    ax.set_xlabel("Mean |SHAP| (predicted probability)")
axes[0, 0].legend(frameon=False, fontsize=7, loc="lower right")
fig.tight_layout(); fig.savefig(O + "fig_shap.png", dpi=300, bbox_inches="tight"); plt.close()

# LIME composite
from PIL import Image
ims = [Image.open(O + f"{d}_lime.png") for d in DS]
w = max(i.width for i in ims); h = max(i.height for i in ims)
canvas = Image.new("RGB", (2 * w, 2 * h), "white")
for k, im in enumerate(ims):
    canvas.paste(im, ((k % 2) * w, (k // 2) * h))
canvas.save(O + "fig_lime.png", dpi=(300, 300))

# ---------------------------------------------------------------- write
json.dump(T, open(O + "tables.json", "w"), indent=1, default=str, ensure_ascii=False)
with pd.ExcelWriter(O + "Revision_Tables.xlsx") as xw:
    for k, v in T.items():
        if isinstance(v, list) and v and isinstance(v[0], list):
            pd.DataFrame(v[1:], columns=v[0]).to_excel(xw, sheet_name=k[:31], index=False)
    for d in DS:
        cand[d].to_excel(xw, sheet_name=f"{d}_dev_folds_candidates", index=False)
        stk[d].to_excel(xw, sheet_name=f"{d}_dev_folds_stacks", index=False)
print("outputs written")
