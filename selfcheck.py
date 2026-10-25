"""Run:  python selfcheck.py
Confirms the dashboard logic reproduces the report: leave-one-scenario-out result (Table 3)
and exact additivity of the Shapley explanations. Takes about 15 seconds."""
import sys
import numpy as np
import slug_core as sc

train = sc.load_training_data('data/training_data.csv')
results = []


def check(name, ok, detail=''):
    results.append(ok)
    print(('PASS  ' if ok else 'FAIL  ') + name + (f'  [{detail}]' if detail else ''))


check('nine scenarios, 2,601 records', train['scenario_id'].nunique() == 9 and len(train) == 2601,
      f"{train['scenario_id'].nunique()} scenarios, {len(train)} records")
check('every scenario has one dominant class of at least 88%',
      all(g['proxy_label'].value_counts(normalize=True).iloc[0] >= 0.88 for _, g in train.groupby('scenario_id')))

for mt in (0.0, 4.1):
    tab = sc.loso_table(train, min_time=mt)
    hgb = int((tab['HistGB'] == tab['dominant_proxy_class']).sum())
    lr = int((tab['LogReg'] == tab['dominant_proxy_class']).sum())
    check(f'leave-one-scenario-out, records after {mt} h: gradient boosting 5 of 9, logistic regression 7 of 9',
          hgb == 5 and lr == 7, f'got {hgb} and {lr}')

models = sc.fit_models(train)
bg = sc.make_background(train)
errs = [sc.explain_record(m, train.iloc[[i]], bg)['additivity_error']
        for m in models.values() for i in (10, 700, 1500, 2400)]
check('Shapley contributions add up exactly (max error below 1e-9)', max(errs) < 1e-9, f'{max(errs):.1e}')

print('\nALL CHECKS PASSED' if all(results) else '\nSOME CHECKS FAILED (a different scikit-learn version can shift individual results slightly)')
sys.exit(0 if all(results) else 1)
