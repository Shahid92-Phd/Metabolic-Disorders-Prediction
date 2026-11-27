import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(10, 7.2))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
INK, DEV, TST, OUT = "#1f2933", "#dbe9f6", "#fbe3d6", "#e3f1df"


def box(x, y, w, h, text, fc, fs=8.5, bold=False, ec=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=INK, wrap=True)


def arrow(x1, y1, x2, y2, ls="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color=INK, linestyle=ls))


box(30, 92, 40, 6, "Four independent public datasets\n(obesity, diabetes, metabolic syndrome, hypertension)", "#ffffff", 8.5, True)
box(25, 82, 50, 5.5, "Stratified 70:30 split (seed 42); physiologically impossible\nzeros recoded as missing before splitting", "#ffffff")
arrow(50, 92, 50, 88)
# development side
ax.add_patch(FancyBboxPatch((2, 18), 62, 60, boxstyle="round,pad=0.4,rounding_size=1.5",
                            fc="#f5f9fd", ec="#4a78a8", lw=1.2, ls="--"))
ax.text(4, 75.5, "Development set (70%) - all selection decisions", fontsize=9, fontweight="bold", color="#24527a")
box(5, 64, 56, 8.5, "Fold-wise pipeline, fitted on the training part of every fold only:\n"
    "outlier capping (obesity) -> imputation -> Yeo-Johnson + z-score -> one-hot\n"
    "-> SMOTE (diabetes, MetS, hypertension) -> classifier", DEV)
box(5, 51, 56, 9, "Nested CV of 12 candidate learners\nouter 5-fold (performance estimate) x inner 3-fold grid search (tuning)\n"
    "ranking by mean outer-fold MCC", DEV)
box(5, 39, 56, 8, "Top-6 learners -> stacking (5-fold out-of-fold meta-features,\nlogistic-regression meta-learner)", DEV)
box(5, 21, 27, 14, "Ablation and statistics\non outer folds only:\ncomposition, tuning, SMOTE,\nin-fold RFE; Friedman +\ncorrected resampled t-tests", DEV, 8)
box(34, 21, 27, 14, "Final refit on the whole\ndevelopment set:\ninner 5-fold re-tuning of\nthe six base learners,\nstack refitted once", DEV, 8)
arrow(33, 64, 33, 60.5); arrow(33, 51, 33, 47.5); arrow(20, 39, 18.5, 35.5); arrow(46, 39, 47.5, 35.5)
arrow(40, 82, 33, 73)
# test side
box(70, 64, 27, 8.5, "Held-out test set (30%)\nnever used for any\nselection or tuning", TST, 8.5, True)
arrow(60, 82, 83.5, 73)
box(70, 40, 27, 19, "Single evaluation of the\nfinal stack and the six\nre-fitted base learners:\naccuracy, sensitivity, specificity,\nPPV, NPV, F1, MCC, kappa,\nROC-AUC, PR-AUC, Brier,\ncalibration, decision curves\n(2000 bootstrap 95% CIs)", TST, 7.8)
arrow(83.5, 64, 83.5, 59.5); arrow(61, 28, 70, 46)
box(70, 18, 27, 18, "Sensitivity analyses\n(reported, not used for\nselection): label-defining\npredictors removed;\nmean/KNN/iterative imputation;\nno outlier capping; original\nleaky diabetes file", TST, 7.8)
arrow(83.5, 40, 83.5, 36.5)
box(10, 3, 80, 10, "Explainability of the complete fitted stack: permutation SHAP (global), fold-to-fold SHAP stability,\n"
    "grouped permutation importance for correlated predictors, meta-learner contributions, LIME (one case);\n"
    "feature-domain recurrence defined as a domain with at least one predictor in the SHAP top five", OUT, 8)
arrow(47.5, 21, 47.5, 13.5); arrow(83.5, 18, 75, 13.5)
plt.savefig("../outputs/fig_workflow.png", dpi=300, bbox_inches="tight")
