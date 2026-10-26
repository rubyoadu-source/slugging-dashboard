"""
SlugGuard - explainable decision support for slugging risk (research prototype).

What it does : classifies the CURRENT record into a proxy flow-stability class with two
               models, explains the flag (exact Shapley values), and lists engineering
               checks and options for an engineer to review.
What it is NOT: a forecast, a validated slugging detector, or a control system.
Run locally  : streamlit run app.py
"""
import io
import os
import tempfile
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import slug_core as sc
from consolidate import consolidate_scenarios
from ingest import load_workbook_scenarios

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE_PATH = os.path.join(HERE, 'artifacts', 'models_full.joblib')

st.set_page_config(page_title='SlugGuard | Flow assurance decision support',
                   page_icon='🌊', layout='wide')


def list_repo_files(limit=80):
    out = []
    for root, dirs, files in os.walk(HERE):
        dirs[:] = sorted(d for d in dirs if d not in ('__pycache__', '.git'))
        for f in sorted(files):
            out.append(os.path.relpath(os.path.join(root, f), HERE))
    return out[:limit]


def find_training_csv():
    """data/training_data.csv if it is there; otherwise any file that looks like it
    (Linux paths are case-sensitive and the GitHub web uploader can flatten folders)."""
    exact = os.path.join(HERE, 'data', 'training_data.csv')
    if os.path.exists(exact):
        return exact
    hits = []
    for root, dirs, files in os.walk(HERE):
        dirs[:] = [d for d in dirs if d not in ('__pycache__', '.git')]
        for f in files:
            name = f.lower().replace(' ', '')
            if name.startswith('training') and name.endswith('.csv'):
                hits.append(os.path.join(root, f))
    return sorted(hits, key=len)[0] if hits else None


DATA_PATH = find_training_csv()
if DATA_PATH is None or os.path.getsize(DATA_PATH) < 1000:
    st.error('The training data file was not found (or is empty): expected `data/training_data.csv` '
             'in the repository, next to `app.py`.')
    st.markdown('Files the app can see in the repository:')
    st.code('\n'.join(list_repo_files()))
    st.markdown('Check that the folder is called exactly `data` (lower case) and holds `training_data.csv`, '
                'then reboot the app from the Streamlit menu.')
    st.stop()

CLASS_COL = {sc.CLASS_SEVERE: '#e5484d', sc.CLASS_STABLE: '#30a46c', sc.CLASS_TRANS: '#f5a524'}
CLASS_RGBA = {sc.CLASS_SEVERE: 'rgba(229,72,77,0.55)', sc.CLASS_STABLE: 'rgba(48,164,108,0.55)',
              sc.CLASS_TRANS: 'rgba(245,165,36,0.55)'}
TEAL = '#2dd4bf'

st.markdown("""
<style>
.block-container {padding-top: 1.0rem; padding-bottom: 2rem;}
.sg-banner {background:#162338;border:1px solid #243752;border-radius:10px;padding:.55rem .9rem;
            color:#b9c7da;font-size:.82rem;letter-spacing:.03em;margin-bottom:.8rem;}
.sg-card {background:#111c2e;border:1px solid #1f2d44;border-radius:12px;padding:.7rem .9rem;height:100%;}
.sg-label {color:#8ea0b8;font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;}
.sg-value {font-size:1.3rem;font-weight:700;margin-top:.1rem;line-height:1.25;}
.sg-sub {color:#8ea0b8;font-size:.78rem;margin-top:.1rem;}
.sg-priority {border-radius:12px;padding:.7rem 1rem;margin:.7rem 0 .4rem 0;font-weight:600;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- cached resources
@st.cache_data(show_spinner=False)
def load_train():
    return sc.load_training_data(DATA_PATH)


TRAIN = load_train()
STATS = sc.reference_stats(TRAIN)
SCEN_DESC = {sid: sc.describe_scenario(g) for sid, g in TRAIN.groupby('scenario_id')}


@st.cache_resource(show_spinner='Preparing the models (first load only)...')
def get_models(held_out):
    """held_out=None: all scenarios; otherwise that scenario is left out of training."""
    if held_out is None:
        loaded = sc.load_bundle(BUNDLE_PATH)
        if loaded is not None:
            return loaded[0], loaded[1], 'Colab export (artifacts/models_full.joblib)'
        d = TRAIN
    else:
        d = TRAIN[TRAIN['scenario_id'] != held_out]
    return sc.fit_models(d), sc.make_background(d), 'trained in the app from data/training_data.csv'


@st.cache_data(show_spinner=False)
def run_predictions(X, held_out):
    models, _, _ = get_models(held_out)
    return sc.predict_all(models, X)


@st.cache_data(show_spinner=False)
def explain_cached(row, held_out, model_name):
    models, bg, _ = get_models(held_out)
    return sc.explain_record(models[model_name], row, bg)


@st.cache_data(show_spinner=False)
def global_cached(sample, held_out, class_name):
    models, bg, _ = get_models(held_out)
    return sc.global_importance(models['HistGB'], sample, bg.sample(n=min(30, len(bg)), random_state=42), class_name)


@st.cache_data(show_spinner=False)
def read_workbooks(items):
    raw = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, content in items:
            path = os.path.join(tmp, os.path.basename(name))
            with open(path, 'wb') as f:
                f.write(content)
            raw += load_workbook_scenarios(path)
    return consolidate_scenarios(raw, reference_median_max_abs=sc.reference_median_max_abs(TRAIN))


@st.cache_data(show_spinner=False)
def loso_cached(min_time):
    return sc.loso_table(TRAIN, min_time=min_time)


# ---------------------------------------------------------------- small helpers
def card(label, value, sub='', colour=None):
    style = f' style="color:{colour}"' if colour else ''
    return (f'<div class="sg-card"><div class="sg-label">{label}</div>'
            f'<div class="sg-value"{style}>{value}</div><div class="sg-sub">{sub}</div></div>')


def normalise_columns(df):
    mapping = {}
    lower = {t.lower(): t for t in sc.FEATURES}
    for c in df.columns:
        k = str(c).strip().lower().replace(' ', '_')
        if k in lower:
            mapping[c] = lower[k]
        elif k in ('time', 'time_h', 'time_(h)', 'time1', 't_h'):
            mapping[c] = 'TIME_h'
    out = df.rename(columns=mapping)
    return out.loc[:, ~out.columns.duplicated()]


def settings_widgets(gas_lift_known, choke_known):
    gl, tcv = None, None
    if not gas_lift_known:
        gl = st.sidebar.number_input('Gas-lift rate (MMSCFD)', 0.0, 30.0, 5.0, 1.0)
    if not choke_known:
        restricted = st.sidebar.radio('Topside choke', ['Unrestricted', 'Restricted'], horizontal=True)
        if restricted == 'Restricted':
            tcv = st.sidebar.number_input('Choke opening (% open)', 1.0, 100.0, 10.0, 1.0)
    return gl, tcv


# ---------------------------------------------------------------- sidebar: data source
st.sidebar.markdown('### 🌊 SlugGuard')
source = st.sidebar.radio('Data source', ['Demo scenarios', 'Upload CSV', 'Upload OLGA workbook'])

held_out, df, label, notes = None, None, '', []

if source == 'Demo scenarios':
    sid = st.sidebar.selectbox('Scenario', list(SCEN_DESC.keys()), format_func=lambda s: SCEN_DESC[s])
    view_mode = st.sidebar.radio('Model view', ['Honest: never saw this scenario',
                                                'In-sample: trained on all nine'])
    held_out = sid if view_mode.startswith('Honest') else None
    df = TRAIN[TRAIN['scenario_id'] == sid].sort_values('TIME_h').reset_index(drop=True)
    label = SCEN_DESC[sid]

elif source == 'Upload CSV':
    up = st.sidebar.file_uploader('CSV with QLT, UL, UG (optional: TIME_h, VALVDP)', type=['csv'])
    if up is None:
        st.info('Upload a CSV with columns QLT (m³/s), UL (m/s), UG (m/s). TIME_h (hours) and VALVDP (bar) '
                'are optional. Gas-lift rate and choke opening can be columns or sidebar entries.')
        st.stop()
    raw_df = normalise_columns(pd.read_csv(io.BytesIO(up.getvalue())))
    if not all(c in raw_df.columns for c in sc.REQUIRED_SIGNALS):
        st.error('The CSV needs QLT, UL and UG columns.')
        st.stop()
    if 'TIME_h' not in raw_df.columns:
        raw_df['TIME_h'] = np.arange(len(raw_df)) * 0.25
        notes.append('No time column found: assumed 15-minute steps.')
    gl, tcv = settings_widgets('gas_lift_MMSCFD' in raw_df.columns,
                               'topside_choke_TCV_pct_open' in raw_df.columns)
    df = sc.fill_settings(raw_df.sort_values('TIME_h').reset_index(drop=True), gl, tcv)
    label = f'Uploaded CSV: {up.name}'

else:
    ups = st.sidebar.file_uploader('OLGA workbook(s) (.xlsx)', type=['xlsx'], accept_multiple_files=True)
    if not ups:
        st.info('Upload one or more OLGA export workbooks. The reader rebuilds the values from the Excel '
                'formulas, runs the corruption and duplicate checks, and takes gas lift and choke from the case name.')
        st.stop()
    meta, master, qlog = read_workbooks(tuple((u.name, u.getvalue()) for u in ups))
    if len(meta) == 0:
        st.error('No valid scenario found. See the quality log below.')
        st.dataframe(qlog, hide_index=True)
        st.stop()
    with st.sidebar.expander('Quality log of the upload'):
        st.dataframe(qlog[['source_sheet', 'finding', 'action']] if len(qlog) else qlog, hide_index=True)
    ids = list(meta['scenario_id'])
    fmt = {r.scenario_id: f'{r.scenario_id} · {r.well_branch} · {r.gas_lift_condition}'
           + ('' if pd.isna(r.topside_choke_TCV_pct_open) else f' · choke {r.topside_choke_TCV_pct_open:.0f}% open')
           for r in meta.itertuples()}
    sid = st.sidebar.selectbox('Scenario in upload', ids, format_func=lambda s: fmt[s])
    m = meta[meta['scenario_id'] == sid].iloc[0]
    d = master[master['scenario_id'] == sid].sort_values('TIME_h').reset_index(drop=True)
    parsed = m['parse_warning'] is None or pd.isna(m['parse_warning'])
    gl_meta = m['gas_lift_MMSCFD'] if parsed else None
    tcv_meta = None if (not parsed or pd.isna(m['topside_choke_TCV_pct_open'])) else float(m['topside_choke_TCV_pct_open'])
    gl, tcv = settings_widgets(parsed, parsed)
    df = sc.fill_settings(d[[c for c in ['TIME_h', 'QLT', 'UL', 'UG', 'VALVDP'] if c in d.columns]],
                          gl_meta if parsed else gl, tcv_meta if parsed else tcv)
    label = fmt[sid]
    if parsed:
        notes.append('Gas lift and choke were read from the case name.')

if not all(c in df.columns for c in sc.REQUIRED_SIGNALS) or len(df) < 3:
    st.error('Not enough data to show a record.')
    st.stop()

df['proxy_label'] = sc.proxy_label(df)
n = len(df)

# new dataset -> reset the replay
data_id = f'{source}|{label}|{n}|{held_out}|{float(df["QLT"].sum()):.6g}'
if st.session_state.get('data_id') != data_id:
    st.session_state['data_id'] = data_id
    st.session_state['pos'] = 0
    st.session_state['playing'] = False
st.session_state['n'] = n
st.session_state.setdefault('playing', False)
if st.session_state.pop('_advance', False):
    st.session_state['pos'] = min(st.session_state.get('pos', 0) + 1, n - 1)
st.session_state['pos'] = int(min(max(st.session_state.get('pos', 0), 0), n - 1))


def _toggle():
    st.session_state['playing'] = not st.session_state['playing']


def _reset():
    st.session_state['pos'] = 0
    st.session_state['playing'] = False


def _step():
    st.session_state['pos'] = min(st.session_state['pos'] + 1, st.session_state['n'] - 1)


# ---------------------------------------------------------------- sidebar: replay controls
st.sidebar.markdown('---')
st.sidebar.markdown('**Replay (simulated live stream)**')
b1, b2, b3 = st.sidebar.columns(3)
b1.button('⏸ Pause' if st.session_state['playing'] else '▶ Play', on_click=_toggle, key='btn_play')
b2.button('⏮ Reset', on_click=_reset, key='btn_reset')
b3.button('⏭ Step', on_click=_step, key='btn_step')
speed = st.sidebar.select_slider('Speed', options=[0.5, 1, 2, 4], value=1,
                                 format_func=lambda x: f'{x} record/s')
st.sidebar.slider('Record (15-min steps)', 0, n - 1, key='pos')
show_proxy = st.sidebar.checkbox('Show proxy label (reference only)', value=(source == 'Demo scenarios'))
pos = st.session_state['pos']

# ---------------------------------------------------------------- predictions
models, background, model_source = get_models(held_out)
preds = run_predictions(df[sc.FEATURES], held_out)
qual = sc.quality_flags(df, STATS)
row, pr, q = df.iloc[pos], preds.iloc[pos], qual.iloc[pos]
hgb_cls, lr_cls = pr['HistGB_class'], pr['LogReg_class']
priority, p_col = sc.priority_for(hgb_cls, lr_cls, q['status'])
recent = preds['HistGB_class'].iloc[max(0, pos - sc.RECENT_N + 1):pos + 1]
recent_share = float((recent == sc.CLASS_SEVERE).mean())

# ---------------------------------------------------------------- header
st.markdown('## 🌊 SlugGuard · flow assurance decision support')
st.markdown('<div class="sg-banner">🧪 SIMULATION DATA &nbsp;·&nbsp; IDENTIFIES THE <b>CURRENT</b> FLOW STATE '
            '(proxy label), DOES NOT FORECAST &nbsp;·&nbsp; READ-ONLY, NO EQUIPMENT COMMANDS</div>',
            unsafe_allow_html=True)
st.caption(f'**{label}**  ·  record {pos + 1} of {n}  ·  time {row["TIME_h"]:.2f} h')
if source == 'Demo scenarios':
    if held_out is None:
        st.warning('In-sample view: these models were trained on this scenario, so they look better here than '
                   'they will on a new case. Switch to the Honest view for the real picture (report, Table 3).')
    else:
        st.info('Honest view: the models were trained WITHOUT this scenario, as in the leave-one-scenario-out test.')
for msg in sc.scenario_quality_messages(df) + notes:
    st.caption('ℹ️ ' + msg)

# KPI row
k1, k2, k3, k4, k5 = st.columns(5)
k1.markdown(card('Gradient boosting', sc.SHORT[hgb_cls], f'score {pr["HistGB_conf"]:.2f} (uncalibrated)',
                 CLASS_COL[hgb_cls]), unsafe_allow_html=True)
k2.markdown(card('Logistic regression', sc.SHORT[lr_cls], f'score {pr["LogReg_conf"]:.2f} (uncalibrated)',
                 CLASS_COL[lr_cls]), unsafe_allow_html=True)
k3.markdown(card('Model agreement', 'Agree' if pr['agree'] else 'Disagree',
                 'agreement is not proof of correctness', sc.GREEN if pr['agree'] else sc.AMBER),
            unsafe_allow_html=True)
q_col = {'ok': sc.GREEN, 'warning': sc.AMBER, 'insufficient': sc.GREY}[q['status']]
k4.markdown(card('Data quality', {'ok': 'OK', 'warning': 'Warning', 'insufficient': 'Insufficient'}[q['status']],
                 q['note'] or 'inside the training range', q_col), unsafe_allow_html=True)
k5.markdown(card('Severe flags, last ~2 h', f'{recent_share:.0%}', f'last {len(recent)} records (backward-looking)',
                 sc.RED if recent_share >= 0.5 else None), unsafe_allow_html=True)
st.markdown(f'<div class="sg-priority" style="background:{p_col}22;border:1px solid {p_col};color:{p_col}">'
            f'PRIORITY: {priority.upper()}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- trends
def trend_figure():
    d, p = df.iloc[:pos + 1], preds.iloc[:pos + 1]
    rows = [('QLT', 'QLT (m³/s)'), ('UL', 'UL (m/s)'), ('UG', 'UG (m/s)')]
    if df['VALVDP'].notna().any():
        rows.append(('VALVDP', 'VALVDP (bar)'))
    strips = [('HistGB_class', 'GradBoost'), ('LogReg_class', 'LogReg')] + ([('proxy', 'Proxy')] if show_proxy else [])
    n_rows = len(rows) + len(strips)
    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.025,
                        row_heights=[1.0] * len(rows) + [0.22] * len(strips))
    for i, (c, title) in enumerate(rows, start=1):
        fig.add_trace(go.Scatter(x=d['TIME_h'], y=d[c], mode='lines', showlegend=False,
                                 line=dict(color=TEAL, width=1.6)), row=i, col=1)
        fig.update_yaxes(title_text=title, title_font=dict(size=11), row=i, col=1)
    step = float(np.nanmedian(np.diff(df['TIME_h']))) if n > 2 else 0.25
    for j, (col_name, title) in enumerate(strips, start=len(rows) + 1):
        series = d['proxy_label'] if col_name == 'proxy' else p[col_name]
        fig.add_trace(go.Bar(x=d['TIME_h'], y=[1] * len(d), width=step, showlegend=False,
                             marker_color=[CLASS_COL[c] for c in series],
                             hovertext=[sc.SHORT[c] for c in series], hoverinfo='text+x'), row=j, col=1)
        fig.update_yaxes(showticklabels=False, showgrid=False, range=[0, 1], title_text=title,
                         title_font=dict(size=10), row=j, col=1)
    fig.update_xaxes(range=[float(df['TIME_h'].min()), float(df['TIME_h'].max())])
    fig.update_xaxes(title_text='Time (h)', row=n_rows, col=1)
    fig.add_vline(x=float(row['TIME_h']), line_dash='dot', line_color='#9fb3c8', row='all', col='all')
    fig.update_layout(template='plotly_dark', height=150 * len(rows) + 48 * len(strips) + 70, bargap=0,
                      margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)')
    return fig


st.plotly_chart(trend_figure())
st.caption('Shown up to the current record only, as in a live view. Strips: predicted class of each model '
           '(green stable, amber transitional, red severe-slugging signature).')

view = st.radio('Panel', ['Why this flag?', 'Recommendations', 'Model comparison', 'Scenario summary',
                          'Event log', 'About and limits'], horizontal=True, key='view',
                label_visibility='collapsed')

# ---------------------------------------------------------------- panels
if view == 'Why this flag?':
    st.markdown('#### Why did the model say this?')
    cls_choice = st.selectbox('Explain which class?', sc.CLASSES, index=sc.CLASSES.index(hgb_cls),
                              format_func=lambda c: sc.SHORT[c])
    mname = st.radio('Model', ['HistGB', 'LogReg'], horizontal=True,
                     format_func=lambda m: 'Gradient boosting' if m == 'HistGB' else 'Logistic regression')
    exp = explain_cached(df.iloc[[pos]][sc.FEATURES], held_out, mname)
    t = exp['table']
    col = f'phi_{sc.SHORT[cls_choice]}'
    t2 = t.assign(_abs=t[col].abs()).sort_values('_abs', ascending=True)
    fig = go.Figure(go.Bar(x=t2[col], y=t2['feature'], orientation='h',
                           marker_color=[CLASS_COL[sc.CLASS_SEVERE] if v > 0 else TEAL for v in t2[col]],
                           text=[f'{v:+.3f}' for v in t2[col]], textposition='outside'))
    fig.update_layout(template='plotly_dark', height=340, margin=dict(l=10, r=40, t=10, b=10),
                      xaxis_title=f'Contribution to the {sc.SHORT[cls_choice]} score (red pushes up, teal pushes down)',
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig)
    st.caption(f'Average {sc.SHORT[cls_choice]} score over the reference sample: {exp["base"][sc.SHORT[cls_choice]]:.2f}. '
               f'For this record: {exp["fx"][sc.SHORT[cls_choice]]:.2f}. The contributions add up exactly to the '
               f'difference (check: {exp["additivity_error"]:.1e}).')
    top = t.reindex(t[col].abs().sort_values(ascending=False).index).head(3)
    for r in top.itertuples():
        v = getattr(r, col)
        val = 'missing' if pd.isna(r.value) else f'{r.value:.4g} {r.unit}'
        st.markdown(f'- **{r.feature}** = {val} {"raises" if v > 0 else "lowers"} the '
                    f'{sc.SHORT[cls_choice]} score by {abs(v):.2f}')
    shown = t[['feature', 'meaning', 'value', 'unit', col]].rename(columns={col: 'contribution'})
    st.dataframe(shown, hide_index=True)
    st.warning('This explains how the MODEL uses its inputs. It does not show what causes slugging. Gas lift, '
               'choke and well are fixed inside each scenario, so part of what the model learns is scenario '
               'identity, and QLT and UL are part of the proxy label itself (report, section 3.4).')
    with st.expander('Global importance on this scenario (takes a few seconds)'):
        if st.button('Compute', key='btn_global'):
            idx = np.linspace(0, n - 1, min(60, n)).astype(int)
            imp = global_cached(df.iloc[idx][sc.FEATURES].reset_index(drop=True), held_out, cls_choice)
            f2 = go.Figure(go.Bar(x=imp.values[::-1], y=list(imp.index[::-1]), orientation='h', marker_color=TEAL))
            f2.update_layout(template='plotly_dark', height=300, margin=dict(l=10, r=10, t=10, b=10),
                             xaxis_title=f'Mean |contribution| to the {sc.SHORT[cls_choice]} score (gradient boosting)',
                             paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(f2)

elif view == 'Recommendations':
    st.markdown('#### Engineering assessment')
    exp = explain_cached(df.iloc[[pos]][sc.FEATURES], held_out, 'HistGB')
    col = f'phi_{sc.SHORT[hgb_cls]}'
    pushes = exp['table'][exp['table'][col] > 0].sort_values(col, ascending=False)
    drivers = list(zip(pushes['feature'], pushes[col]))
    tcv_val = None if pd.isna(row['topside_choke_TCV_pct_open']) else float(row['topside_choke_TCV_pct_open'])
    ctx = dict(hgb_class=hgb_cls, lr_class=lr_cls, agree=bool(pr['agree']), quality_status=q['status'],
               quality_note=q['note'], gas_lift=None if pd.isna(row['gas_lift_MMSCFD']) else float(row['gas_lift_MMSCFD']),
               tcv_open=tcv_val, valvdp_available=not pd.isna(row['VALVDP']), time_h=float(row['TIME_h']),
               recent_severe_share=recent_share, drivers=drivers)
    rec = sc.recommend(ctx)
    st.markdown(f'<div class="sg-priority" style="background:{rec["colour"]}22;border:1px solid {rec["colour"]};'
                f'color:{rec["colour"]}">{rec["priority"].upper()} · {rec["headline"]}</div>', unsafe_allow_html=True)
    if drivers and rec['priority'] != 'Data insufficient':
        st.markdown('**Main reasons behind the flag:** ' + ', '.join(f'{f} (+{v:.2f})' for f, v in drivers[:3]))
    for nt in rec['notes']:
        st.info(nt)
    st.markdown('**What to check**')
    for c in rec['checks']:
        st.markdown(f'- {c}')
    if rec['options']:
        st.markdown('**Options for the engineer to evaluate (not instructions)**')
        for o in rec['options']:
            st.markdown(f'**{o["action"]}**  \n*Evidence:* {o["evidence"]}  \n*Caveat:* {o["caveat"]}')
    st.warning('Decision-support prompts only. Evidence comes from nine simulated scenarios with a proxy label. '
               'Any real change must follow the approved procedures and be made by the responsible engineer. '
               'This tool never sends commands to equipment.')

elif view == 'Model comparison':
    st.markdown('#### Do the two models agree?')
    d, p = df.iloc[:pos + 1], preds.iloc[:pos + 1]
    c1, c2 = st.columns(2)
    for cc, name, title in ((c1, 'HistGB', 'Gradient boosting'), (c2, 'LogReg', 'Logistic regression')):
        fig = go.Figure()
        for cls in (sc.CLASS_STABLE, sc.CLASS_TRANS, sc.CLASS_SEVERE):
            fig.add_trace(go.Scatter(x=d['TIME_h'], y=p[f'{name}_p_{sc.SHORT[cls]}'], mode='lines', stackgroup='one',
                                     name=sc.SHORT[cls], line=dict(width=0.5, color=CLASS_COL[cls]),
                                     fillcolor=CLASS_RGBA[cls]))
        fig.update_layout(template='plotly_dark', height=260, title=title, margin=dict(l=10, r=10, t=40, b=10),
                          yaxis=dict(range=[0, 1], title='score'), xaxis_title='Time (h)',
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        cc.plotly_chart(fig)
    agree_share = float(p['agree'].mean())
    st.markdown(f'The models agree on **{agree_share:.0%}** of the {len(p)} records shown. '
                'Where they differ, the dashboard asks for engineer review instead of showing false certainty. '
                'In the report, the two models disagreed on several scenarios (SC01, SC05, SC07 in all-record tests), '
                'and neither was clearly better.')
    last = pd.DataFrame({'time_h': d['TIME_h'].tail(10).round(2).values,
                         'gradient boosting': p['HistGB_class'].tail(10).map(sc.SHORT).values,
                         'logistic regression': p['LogReg_class'].tail(10).map(sc.SHORT).values})
    st.dataframe(last, hide_index=True)

elif view == 'Scenario summary':
    st.markdown('#### Whole-scenario summary (offline view of the full record)')
    share = pd.DataFrame({
        'Gradient boosting': preds['HistGB_class'].value_counts(normalize=True),
        'Logistic regression': preds['LogReg_class'].value_counts(normalize=True),
        'Proxy label (reference)': df['proxy_label'].value_counts(normalize=True)}).reindex(sc.CLASSES).fillna(0)
    share.index = [sc.SHORT[c] for c in share.index]
    st.dataframe((share * 100).round(1).rename_axis('share of records (%)'))
    dom = df['proxy_label'].value_counts().index[0]
    for name, col in (('Gradient boosting', 'HistGB_class'), ('Logistic regression', 'LogReg_class')):
        maj = preds[col].value_counts().index[0]
        st.markdown(f'- **{name}**: most common class **{sc.SHORT[maj]}** vs proxy-label dominant class '
                    f'**{sc.SHORT[dom]}** → {"match" if maj == dom else "no match"}')
    if source == 'Demo scenarios' and held_out is None:
        st.warning('In-sample: do not read these matches as performance.')
    st.markdown('---')
    st.markdown('**Leave-one-scenario-out check on the nine demo scenarios** (the headline test of the report). '
                'Each model is trained without the scenario it is tested on.')
    mt = st.radio('Records used', ['All records', 'Only after the first 4.1 h'], horizontal=True)
    if st.button('Run the check (about 10 seconds)', key='btn_loso'):
        tab = loso_cached(0.0 if mt == 'All records' else sc.STARTUP_HOURS)
        show = tab.copy()
        for c in ('dominant_proxy_class', 'HistGB', 'LogReg'):
            show[c] = show[c].map(sc.SHORT)
        st.dataframe(show, hide_index=True)
        for name in ('HistGB', 'LogReg'):
            st.markdown(f'**{name}:** dominant class right for {(tab[name] == tab["dominant_proxy_class"]).sum()} of '
                        f'{len(tab)} scenarios.')

elif view == 'Event log':
    st.markdown('#### Events up to the current record')
    d = pd.DataFrame({'record': np.arange(pos + 1), 'time_h': df['TIME_h'].iloc[:pos + 1].round(2).values,
                      'gradient_boosting': preds['HistGB_class'].iloc[:pos + 1].values,
                      'logistic_regression': preds['LogReg_class'].iloc[:pos + 1].values,
                      'data_quality': qual['status'].iloc[:pos + 1].values})
    d['priority'] = [sc.priority_for(a, b, c)[0] for a, b, c in
                     zip(d['gradient_boosting'], d['logistic_regression'], d['data_quality'])]
    key = d['gradient_boosting'] + '|' + d['logistic_regression'] + '|' + d['priority']
    ev = d[key != key.shift()].copy()
    ev['agreement'] = np.where(ev['gradient_boosting'] == ev['logistic_regression'], 'agree', 'disagree')
    ev['gradient_boosting'] = ev['gradient_boosting'].map(sc.SHORT)
    ev['logistic_regression'] = ev['logistic_regression'].map(sc.SHORT)
    st.caption('A line is added whenever a class, the model agreement or the priority changes.')
    st.dataframe(ev, hide_index=True)
    st.download_button('Download log (CSV)', data=ev.to_csv(index=False).encode('utf-8'),
                       file_name='slugguard_event_log.csv', mime='text/csv')
    st.caption('Engineer review is not recorded in this prototype. It would be added with a persistent store.')

else:
    st.markdown('#### About and limits')
    st.markdown(f"""
**What the classes mean.** *Stable*, *Transitional* and *Severe-Slugging Signature* are a **proxy label** built
from swings in liquid flow and reversals of liquid velocity at the riser outlet (report, equations 3 to 5). They
are not OLGA flow regimes and have not been checked against OLGA's own flow-regime output.

**Identification, not prediction.** Each record is classified on its own. Nothing here forecasts a future event.
The "severe flags, last ~2 h" box looks backwards, not forwards.

**Nine scenarios.** Every scenario is almost one class, so the task is effectively nine scenario-level labels.
On unseen scenarios the logistic regression named the dominant class in 7 of 9 cases and gradient boosting in
5 of 9. Treat any single flag with care.

**Scores are not probabilities.** They are not calibrated.

**Explanations** are exact Shapley values over all 2⁷ = 128 input coalitions against a 100-record reference
sample. They match the SHAP ranking of the Colab notebook, but they describe the model, not physical causes.

**Start-up.** The first hours of each simulated run may reflect start-up, and the recommendation panel says so.

**Models in use:** {model_source}. Training data: `data/training_data.csv` (nine OLGA scenarios, proxy labels).

**Safety.** Read-only research prototype. It must not be connected to control systems or used for operational decisions.
""")

# ---------------------------------------------------------------- replay loop
if st.session_state['playing']:
    if pos < n - 1:
        time.sleep(1.0 / speed)
        st.session_state['_advance'] = True
        st.rerun()
    else:
        st.session_state['playing'] = False
