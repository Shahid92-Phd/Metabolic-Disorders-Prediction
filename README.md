# Analysis: explainable stacking for four cardiometabolic disorders

Code and outputs for the revised manuscript (bioengineering-4622383).

## Data (`data/`)
| File | Source | Notes |
|---|---|---|
| `Obesity_Dataset.xls` (CSV text) | Palechor & Manotas (UCI), 13 predictors | as supplied by the authors |
| `pima.csv` | Pima Indians Diabetes Database, original values | zeros in GL, BP, ST, IN, BMI are recoded as missing in `common.load` |
| `Metabolic Syndrome_Dataset.xls` (CSV text) | NHANES-derived Kaggle dataset | as supplied |
| `fram.csv` | Framingham teaching dataset | outcome `prevalentHyp` (identical to "Risk" in `Hypertension_Dataset.xls`) |
| `Diabetes_Dataset.xls` | earlier file with class-conditional median imputation | used **only** for the leakage sensitivity analysis |

## Run order (Python 3.13; scikit-learn 1.9.1, imbalanced-learn 0.14.2, shap 0.54.1, lime 0.2.0.1)
```
for d in OB DB MS HT; do
  python run_dev.py   $d   # nested CV on the 70% development set; ranking; ablation stacks (no test data)
  python run_final.py $d   # final stack, single test evaluation, bootstrap CIs, calibration, DCA, sensitivity analyses
  python run_xai.py   $d   # SHAP on the full stack, stability, grouped permutation, meta-learner, LIME
  python run_lime.py  $d   # LIME with readable category names (overwrites the LIME part)
done
python make_outputs.py     # all manuscript tables (Revision_Tables.xlsx) and figures
```
`common.py` holds data loading, the in-fold preprocessing pipeline and metric definitions. Random seed 42
throughout. The full run takes about 2 h on two CPU cores.

## Outputs (`outputs/`)
* `<D>_test_predictions.csv` — test-row index in the dataset, true label, predicted label, class probabilities
* `<D>_split_indices.csv` — development/test assignment of every record
* `<D>_dev_candidates.csv`, `<D>_dev_stacks.csv` — outer-fold metrics (nested CV, ablation)
* `<D>_dev_params.json` — hyperparameters selected in each outer fold; `<D>_dev_rfe.json` — in-fold RFE selections
* `<D>_final.json` — final hyperparameters, test metrics with CIs, single-learner comparisons, sensitivity analyses
* `<D>_xai.json` — SHAP importances, stability, grouped importance, meta-learner contributions, LIME
* `Revision_Tables.xlsx` — every table in the revised manuscript; `fig_*.png` — figures

D: OB = obesity, DB = diabetes, MS = metabolic syndrome, HT = hypertension.
