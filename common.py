"""
Shared components for the revised cardiometabolic stacking analysis.

Design (revision):
  * Stratified 70:30 split; the 30 % test set is used once, for final reporting only.
  * Every data-dependent preprocessing step (outlier capping, imputation, Yeo-Johnson,
    standardisation, one-hot encoding, SMOTE, feature selection) is fitted inside the
    training portion of each fold via a single imblearn Pipeline.
  * Hyperparameters are tuned by nested CV on the development (70 %) set.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import PowerTransformer, OneHotEncoder, label_binarize
from sklearn.pipeline import Pipeline as SkPipeline
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (GradientBoostingClassifier, AdaBoostClassifier,
                              RandomForestClassifier, ExtraTreesClassifier)
from sklearn import metrics as M

warnings.filterwarnings("ignore")
SEED = 42
import os
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "")

# --------------------------------------------------------------------------- data
OB_CLASSES = ["Insufficient_Weight", "Normal_Weight", "Overweight_Level_I",
              "Overweight_Level_II", "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"]
OB_LABELS = ["Underweight", "Normal weight", "Overweight I", "Overweight II",
             "Obesity I", "Obesity II", "Obesity III"]


def load(name):
    """Return X (raw, with NaN for missing), y (int), spec dict."""
    if name == "OB":
        d = pd.read_csv(DATA + "Obesity_Dataset.xls")
        for c in ["FH", "FF"]:
            d[c] = d[c].map({"no": 0, "yes": 1})
        ordmap = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
        for c in ["CM", "CA"]:
            d[c] = d[c].map(ordmap)
        y = d.pop("OB").map({c: i for i, c in enumerate(OB_CLASSES)}).astype(int)
        spec = dict(cont=["AG", "HT", "WT", "FV", "NM", "CM", "CW", "FA", "TD", "CA"],
                    binary=["FH", "FF"], nominal=["TU"], cap=["AG", "NM"],
                    label_feats=["HT", "WT"], smote=False, classes=OB_LABELS,
                    title="Obesity")
    elif name == "DB":
        cols = ["PR", "GL", "BP", "ST", "IN", "BMI", "DPF", "AG", "DB"]
        d = pd.read_csv(DATA + "pima.csv", header=None, names=cols)
        for c in ["GL", "BP", "ST", "IN", "BMI"]:   # physiologically impossible zeros = missing
            d.loc[d[c] == 0, c] = np.nan
        y = d.pop("DB").astype(int)
        spec = dict(cont=["PR", "GL", "BP", "ST", "IN", "BMI", "DPF", "AG"], binary=[], nominal=[],
                    cap=[], label_feats=["GL"], smote=True, classes=["No diabetes", "Diabetes"],
                    title="Diabetes")
    elif name == "MS":
        d = pd.read_csv(DATA + "Metabolic Syndrome_Dataset.xls")
        d["GD"] = d["GD"].map({"Female": 0, "Male": 1})
        y = d.pop("MS").astype(int)
        spec = dict(cont=["AG", "IL", "WCM", "BMI", "UC", "UA", "BGL", "HDL", "TL"],
                    binary=["GD"], nominal=["MT", "EB"], cap=[],
                    label_feats=["WCM", "BGL", "HDL", "TL"], smote=True,
                    classes=["No MetS", "MetS"], title="Metabolic syndrome")
    elif name == "HT":
        f = pd.read_csv(DATA + "fram.csv")
        keep = ["male", "age", "currentSmoker", "cigsPerDay", "BPMeds", "diabetes", "totChol",
                "sysBP", "diaBP", "BMI", "heartRate", "glucose"]
        d = f[keep].copy()
        y = f["prevalentHyp"].astype(int)   # identical to 'Risk' in the original file
        spec = dict(cont=["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"],
                    binary=["male", "currentSmoker", "BPMeds", "diabetes"], nominal=[], cap=[],
                    label_feats=["sysBP", "diaBP", "BPMeds"], smote=True,
                    classes=["No hypertension", "Hypertension"], title="Hypertension")
    else:
        raise ValueError(name)
    return d.reset_index(drop=True), y.reset_index(drop=True), spec


# ------------------------------------------------------------------ preprocessing
class SDCapper(BaseEstimator, TransformerMixin):
    """Cap selected columns at mean +/- 3 SD learned on the training fold (original notebook rule)."""
    def __init__(self, idx=()):
        self.idx = idx

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_, self.hi_ = {}, {}
        for i in self.idx:
            m, s = np.nanmean(X[:, i]), np.nanstd(X[:, i], ddof=1)
            self.lo_[i], self.hi_[i] = m - 3 * s, m + 3 * s
        return self

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)

    def transform(self, X):
        X = np.array(X, dtype=float, copy=True)
        for i in self.idx:
            X[:, i] = np.clip(X[:, i], self.lo_[i], self.hi_[i])
        return X


def make_imputer(kind, strategy="median"):
    if kind == "median":
        return SimpleImputer(strategy=strategy)
    if kind == "mean":
        return SimpleImputer(strategy="mean")
    if kind == "knn":
        return KNNImputer(n_neighbors=5)
    if kind == "iterative":
        return IterativeImputer(max_iter=20, random_state=SEED, sample_posterior=False)
    raise ValueError(kind)


def make_pre(spec, cols, impute="median", cap=True):
    cont = [c for c in spec["cont"] if c in cols]
    binary = [c for c in spec["binary"] if c in cols]
    nominal = [c for c in spec["nominal"] if c in cols]
    steps = []
    capidx = [cont.index(c) for c in spec["cap"] if c in cont] if cap else []
    if capidx:
        steps.append(("cap", SDCapper(tuple(capidx))))
    steps += [("imp", make_imputer(impute)),
              ("yj", PowerTransformer(method="yeo-johnson", standardize=True))]
    tr = []
    if cont:
        tr.append(("cont", SkPipeline(steps), cont))
    if binary:
        tr.append(("bin", SimpleImputer(strategy="most_frequent"), binary))
    if nominal:
        tr.append(("nom", SkPipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                      ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
                   nominal))
    return ColumnTransformer(tr, remainder="drop", verbose_feature_names_out=False)


# ---------------------------------------------------------------------- learners
def learners():
    """name -> (estimator, compact grid). Grids are kept small for nested CV."""
    return {
        "NB": (GaussianNB(), {"var_smoothing": [1e-9, 1e-6, 1e-3]}),
        "SVM": (SVC(probability=True, random_state=SEED), {"C": [0.1, 1, 10], "gamma": ["scale", 0.05]}),
        "QDA": (QuadraticDiscriminantAnalysis(), {"reg_param": [0.0, 0.1, 0.5]}),
        "Ridge": (RidgeClassifier(), {"alpha": [0.1, 1.0, 10.0]}),
        "KNN": (KNeighborsClassifier(), {"n_neighbors": [5, 11, 21], "weights": ["uniform", "distance"]}),
        "LDA": (LinearDiscriminantAnalysis(solver="lsqr"), {"shrinkage": [None, "auto"]}),
        "LR": (LogisticRegression(max_iter=3000), {"C": [0.01, 0.1, 1.0, 10.0]}),
        "DT": (DecisionTreeClassifier(random_state=SEED),
               {"max_depth": [3, 5, 8, None], "min_samples_leaf": [1, 5, 10]}),
        "GB": (GradientBoostingClassifier(n_estimators=150, random_state=SEED),
               {"learning_rate": [0.05, 0.1], "max_depth": [2, 3]}),
        "ADB": (AdaBoostClassifier(random_state=SEED),
                {"n_estimators": [100, 300], "learning_rate": [0.1, 0.5, 1.0]}),
        "RF": (RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=1),
               {"max_features": ["sqrt", 0.5], "min_samples_leaf": [1, 3]}),
        "ET": (ExtraTreesClassifier(n_estimators=200, random_state=SEED, n_jobs=1),
               {"max_features": ["sqrt", 0.5], "min_samples_leaf": [1, 3]}),
    }


def make_pipe(spec, cols, est, smote=None, impute="median", cap=True):
    smote = spec["smote"] if smote is None else smote
    steps = [("pre", make_pre(spec, cols, impute, cap))]
    if smote:
        steps.append(("smote", SMOTE(random_state=SEED, k_neighbors=5)))
    steps.append(("clf", clone(est)))
    return Pipeline(steps)


# ----------------------------------------------------------------------- metrics
def scores_of(model, X):
    """Probability-like scores: (n,) for binary positive class, (n,K) for multiclass."""
    if hasattr(model, "predict_proba"):
        try:
            p = model.predict_proba(X)
            return p[:, 1] if p.shape[1] == 2 else p
        except Exception:
            pass
    s = model.decision_function(X)
    return s


def has_proba(model):
    try:
        return hasattr(model, "predict_proba") and not isinstance(
            model.named_steps["clf"] if hasattr(model, "named_steps") else model, RidgeClassifier)
    except Exception:
        return hasattr(model, "predict_proba")


def ece_binary(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e


def cal_slope_intercept(y, p):
    from sklearn.linear_model import LogisticRegression as LR_
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p)).reshape(-1, 1)
    slope = LR_(C=1e6, max_iter=1000).fit(lp, y).coef_[0, 0]
    # calibration-in-the-large: intercept with slope fixed at 1 (offset)
    from scipy.optimize import brentq
    lpr = lp.ravel(); yy = np.asarray(y, float)
    inter = brentq(lambda a: np.sum(yy - 1 / (1 + np.exp(-(a + lpr)))), -20, 20)
    return slope, inter


def compute_metrics(y, pred, score, K, proba=True):
    """Positive class = 1 for binary tasks. Multiclass metrics are macro-averaged."""
    y = np.asarray(y); pred = np.asarray(pred)
    out = {"Accuracy": M.accuracy_score(y, pred),
           "Balanced accuracy": M.balanced_accuracy_score(y, pred),
           "Kappa": M.cohen_kappa_score(y, pred),
           "MCC": M.matthews_corrcoef(y, pred)}
    if K == 2:
        tn, fp, fn, tp = M.confusion_matrix(y, pred, labels=[0, 1]).ravel()
        out.update({"Precision (PPV)": tp / (tp + fp) if tp + fp else np.nan,
                    "Recall (sensitivity)": tp / (tp + fn),
                    "Specificity": tn / (tn + fp),
                    "NPV": tn / (tn + fn) if tn + fn else np.nan,
                    "F1": M.f1_score(y, pred)})
        out["ROC-AUC"] = M.roc_auc_score(y, score)
        out["PR-AUC"] = M.average_precision_score(y, score) if proba else np.nan
        if proba:
            out["Brier"] = M.brier_score_loss(y, score)
            out["ECE"] = ece_binary(y, score)
    else:
        out.update({"Precision (PPV)": M.precision_score(y, pred, average="macro", zero_division=0),
                    "Recall (sensitivity)": M.recall_score(y, pred, average="macro"),
                    "F1": M.f1_score(y, pred, average="macro"),
                    "Quadratic-weighted kappa": M.cohen_kappa_score(y, pred, weights="quadratic")})
        if proba and score.ndim == 2 and np.allclose(score.sum(1), 1, atol=1e-3):
            Y = label_binarize(y, classes=range(K))
            out["ROC-AUC"] = M.roc_auc_score(y, score, multi_class="ovr", average="macro")
            out["PR-AUC"] = M.average_precision_score(Y, score, average="macro")
            out["Brier"] = np.mean(np.sum((score - Y) ** 2, axis=1)) / 2.0  # scaled to [0,1]
        else:
            out["ROC-AUC"] = np.nan
    return out


def mcc_scorer():
    return M.make_scorer(M.matthews_corrcoef)
