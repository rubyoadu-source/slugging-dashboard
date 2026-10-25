"""
consolidate.py — turns raw sheets into clean, de-duplicated scenarios.

Automatic checks (no hard-coded sheet names):
 1. a sheet is corrupted if two different physical columns are (almost) identical;
 2. a sheet is corrupted if a column's maximum is more than 20x the typical maximum
    for that column (typical = median over all loaded sheets, or over a supplied
    reference such as the training data);
 3. sheets with the same .genkey are duplicates (kept once) or are merged column-wise
    when they hold different variables (e.g. pressure export of the same case).
"""
import re
import pandas as pd
import numpy as np

GENKEY_PATTERN = re.compile(
    r'Report-(?P<year>\d{4})_Oil_(?P<oil>\d+)bopd_Liquid_(?P<liquid>\d+)blpd_'
    r'(?:(?P<gl_rate>\d+)MMSCFD_GL|No_GL)_TD_Only_(?P<well>OP\d)'
    r'(?:_TCV_(?P<tcv>\d+)pc)?'
)


def parse_genkey(genkey):
    g = genkey.replace('.genkey', '')
    m = GENKEY_PATTERN.search(g)
    if not m:
        return {'design_year': None, 'oil_rate_bopd': None, 'liquid_rate_blpd': None,
                'well_branch': None, 'gas_lift_MMSCFD': None, 'gas_lift_condition': None,
                'topside_choke_TCV_pct_open': None, 'parse_warning': f'Could not parse: {genkey}'}
    d = m.groupdict()
    gas_lift = 0 if d['gl_rate'] is None else int(d['gl_rate'])
    return {'design_year': int(d['year']), 'oil_rate_bopd': int(d['oil']),
            'liquid_rate_blpd': int(d['liquid']), 'well_branch': d['well'],
            'gas_lift_MMSCFD': gas_lift,
            'gas_lift_condition': 'No Gas Lift' if gas_lift == 0 else f'{gas_lift} MMSCFD Gas Lift',
            'topside_choke_TCV_pct_open': int(d['tcv']) if d['tcv'] else None,
            'parse_warning': None}


def detect_internal_corruption(df, value_cols, identical_frac_threshold=0.95, corr_threshold=0.999):
    cols = [c for c in value_cols if c in df.columns and df[c].notna().sum() > 5]
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            both = pd.concat([df[cols[i]], df[cols[j]]], axis=1).dropna()
            if len(both) < 5:
                continue
            frac_equal = (both.iloc[:, 0] == both.iloc[:, 1]).mean()
            if frac_equal >= identical_frac_threshold:
                return True, (f"Columns '{cols[i]}' and '{cols[j]}' are identical in "
                              f"{frac_equal:.0%} of rows - not physically plausible.")
            corr = both.iloc[:, 0].corr(both.iloc[:, 1])
            if pd.notna(corr) and corr >= corr_threshold and both.iloc[:, 0].std() > 0:
                ratio = (both.iloc[:, 0] / both.iloc[:, 1].replace(0, np.nan)).median()
                if pd.notna(ratio) and abs(ratio - 1) < 0.01:
                    return True, (f"Columns '{cols[i]}' and '{cols[j]}' are near-identical "
                                  f"(corr={corr:.4f}) - flagged as corrupted export.")
    return False, None


def detect_magnitude_outlier(df, value_cols, typical_max_abs, multiplier=20.0):
    for c in value_cols:
        if c not in df.columns or c not in typical_max_abs:
            continue
        med = typical_max_abs[c]
        if med is None or med == 0 or pd.isna(med):
            continue
        this_max = df[c].abs().max()
        if pd.notna(this_max) and this_max > multiplier * med:
            return True, (f"Column '{c}' max |value| = {this_max:.3g}, more than {multiplier:.0f}x the "
                          f"typical maximum ({med:.3g}) - likely a corrupted export.")
    return False, None


def consolidate_scenarios(raw_scenarios, value_cols=('QLT', 'UL', 'UG'), reference_median_max_abs=None):
    """Returns (metadata_df, master_df, quality_log_df).
    reference_median_max_abs: optional {column: typical max|value|} (e.g. from the
    training data) so that a single uploaded sheet can still be checked."""
    quality_log = []
    by_genkey = {}
    for s in raw_scenarios:
        by_genkey.setdefault(s['genkey'], []).append(s)

    typical = {}
    for c in value_cols:
        maxes = [s['df'][c].abs().max() for s in raw_scenarios
                 if c in s['df'].columns and s['df'][c].notna().any()]
        typical[c] = float(np.median(maxes)) if maxes else None
    if reference_median_max_abs:
        typical.update({k: v for k, v in reference_median_max_abs.items() if k in value_cols})

    scenario_id = 0
    metadata_rows, master_frames = [], []

    for genkey, group in by_genkey.items():
        checked = []
        for s in group:
            id_bad, id_reason = detect_internal_corruption(s['df'], list(value_cols))
            mag_bad, mag_reason = detect_magnitude_outlier(s['df'], list(value_cols), typical)
            bad = id_bad or mag_bad
            reason = ' AND '.join(r for r in (id_reason, mag_reason) if r) if bad else None
            checked.append((s, bad, reason))
            if bad:
                quality_log.append({'genkey': genkey, 'source_file': s['source_file'],
                                    'source_sheet': s['source_sheet'], 'finding': reason,
                                    'action': 'Excluded (corrupted)'})

        valid = [c for c in checked if not c[1]]
        if not valid:
            quality_log.append({'genkey': genkey, 'source_file': 'ALL COPIES', 'source_sheet': '-',
                                'finding': 'Every copy of this case failed the corruption check.',
                                'action': 'EXCLUDED - no valid copy available'})
            continue

        primary = valid[0][0]
        for s, bad, reason in checked:
            if s is primary or bad:
                continue
            overlap = (set(s['df'].columns) & set(primary['df'].columns)) - {'TIME_h'}
            if overlap:
                quality_log.append({'genkey': genkey, 'source_file': s['source_file'],
                                    'source_sheet': s['source_sheet'],
                                    'finding': f"Duplicate export of {sorted(overlap)}.",
                                    'action': 'Excluded (redundant duplicate)'})
            else:
                quality_log.append({'genkey': genkey, 'source_file': s['source_file'],
                                    'source_sheet': s['source_sheet'],
                                    'finding': f"New columns {sorted(set(s['df'].columns) - {'TIME_h'})} merged.",
                                    'action': 'Merged'})
                n = min(len(primary['df']), len(s['df']))
                for c in [c for c in s['df'].columns if c != 'TIME_h']:
                    vals = np.full(len(primary['df']), np.nan)
                    vals[:n] = s['df'][c].values[:n]
                    primary['df'][c] = vals

        scenario_id += 1
        sid = f"SC{scenario_id:02d}"
        meta = parse_genkey(genkey)
        meta.update({'scenario_id': sid, 'genkey': genkey, 'source_file': primary['source_file'],
                     'source_sheet': primary['source_sheet'], 'n_timesteps': len(primary['df']),
                     'n_duplicate_copies_found': len(group)})
        metadata_rows.append(meta)
        df = primary['df'].copy()
        df.insert(0, 'scenario_id', sid)
        master_frames.append(df)

    metadata = pd.DataFrame(metadata_rows)
    master = pd.concat(master_frames, ignore_index=True) if master_frames else pd.DataFrame()
    return metadata, master, pd.DataFrame(quality_log)
