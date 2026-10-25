# =====================================================================================
# PLACE: a NEW CELL at the very END of Slugging_ML_Pipeline.ipynb
#        (below the scenario-level evaluation cell).
# NEEDS IN MEMORY (all created earlier in the notebook): feat, y, X_cols_7
# RUN IT: after any change to the data or the label, e.g. when you add scenarios.
# WHAT IT DOES: writes the two files the dashboard repository needs and downloads them:
#   training_data.csv   -> put it in  slugging-dashboard/data/
#   models_full.joblib  -> (optional) put it in  slugging-dashboard/artifacts/
# =====================================================================================
import numpy as np, pandas as pd, sklearn, joblib
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ['QLT', 'UL', 'UG', 'VALVDP', 'gas_lift_MMSCFD', 'topside_choke_TCV_pct_open', 'is_choked']
assert list(X_cols_7) == FEATURES, f'X_cols_7 {list(X_cols_7)} differs from the dashboard list {FEATURES}'
assert len(y) == len(feat), 'y and feat are not aligned'

out = feat[['scenario_id', 'TIME_h', 'well_branch'] + FEATURES].copy()
out['proxy_label'] = np.asarray(y)
out = out.sort_values(['scenario_id', 'TIME_h']).reset_index(drop=True)
out.to_csv('training_data.csv', index=False)

models = {
    'HistGB': HistGradientBoostingClassifier(random_state=42, max_iter=200),
    'LogReg': Pipeline([('imp', SimpleImputer(strategy='median', add_indicator=True)),
                        ('sc', StandardScaler()),
                        ('lr', LogisticRegression(max_iter=2000, class_weight='balanced'))]),
}
fitted = {k: clone(m).fit(out[FEATURES], out['proxy_label']) for k, m in models.items()}
background = out[FEATURES].sample(n=min(100, len(out)), random_state=42).reset_index(drop=True)
joblib.dump({'models': fitted, 'background': background, 'features': FEATURES,
             'classes': sorted(out['proxy_label'].unique()), 'sklearn_version': sklearn.__version__},
            'models_full.joblib')

print(f"scenarios: {out['scenario_id'].nunique()} | rows: {len(out)} | scikit-learn {sklearn.__version__}")
print(out['proxy_label'].value_counts().to_string())
try:
    from google.colab import files
    files.download('training_data.csv')
    files.download('models_full.joblib')
except ImportError:
    print('Not in Colab: the files are in the current folder.')
