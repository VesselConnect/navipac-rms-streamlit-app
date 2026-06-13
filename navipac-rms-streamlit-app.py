import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re
import io
import zipfile
from datetime import datetime, time

st.set_page_config(page_title='NaviPac Motion RMS', layout='wide')
st.title('NaviPac Motion RMS Analyser')

# ---------------------------------------------------------------------------
# PARSER  – handles comma-delimited NaviPac .NPD export
# Line format:  dd.mm.yyyy HH:MM:SS, heading, roll, pitch, heave, HC
# ---------------------------------------------------------------------------

# Regex: date  time  heading  roll  pitch  heave  (optional HC flag)
LINE_RE = re.compile(
    r'^(\d{2}\.\d{2}\.\d{4})\s+(\d{2}:\d{2}:\d{2})'
    r'\s*,\s*([\d.eE+\-]+)'   # heading
    r'\s*,\s*([\d.eE+\-]+)'   # roll
    r'\s*,\s*([\d.eE+\-]+)'   # pitch
    r'\s*,\s*([\d.eE+\-]+)'   # heave
    r'(?:\s*,\s*HC)?'
)


def parse_npd(file_bytes: bytes, filename: str) -> pd.DataFrame:
    text = file_bytes.decode('latin-1', errors='replace')
    rows = []
    for line in text.splitlines():
        m = LINE_RE.match(line.strip())
        if m:
            date_str, time_str, hdg, roll, pitch, heave = m.groups()
            try:
                dt = datetime.strptime(f'{date_str} {time_str}', '%d.%m.%Y %H:%M:%S')
                rows.append({
                    'datetime': dt,
                    'heading': float(hdg),
                    'roll':    float(roll),
                    'pitch':   float(pitch),
                    'heave_m': float(heave),
                })
            except ValueError:
                pass
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df['source'] = filename
    return df


def rms(series: pd.Series) -> float:
    return float(np.sqrt((series ** 2).mean())) if len(series) else np.nan


def heading_in_range(hdg: pd.Series, center: float, tol: float) -> pd.Series:
    """Return boolean mask for heading within ±tol of center (wraps 0/360)."""
    diff = ((hdg - center + 180) % 360) - 180
    return diff.abs() <= tol


# ---------------------------------------------------------------------------
# SIDEBAR – controls
# ---------------------------------------------------------------------------
st.sidebar.header('Settings')

uploaded = st.sidebar.file_uploader(
    'Upload .NPD file(s)', type=['NPD', 'npd', 'txt'], accept_multiple_files=True
)

if not uploaded:
    st.info('Upload one or more NaviPac .NPD files using the sidebar.')
    st.stop()

# Parse all files
all_dfs = []
for f in uploaded:
    df_f = parse_npd(f.read(), f.name)
    if not df_f.empty:
        all_dfs.append(df_f)
    else:
        st.warning(f'No parseable rows found in **{f.name}**')

if not all_dfs:
    st.error('No data could be parsed from the uploaded files.')
    st.stop()

df_all = pd.concat(all_dfs, ignore_index=True).sort_values('datetime')

# ---------------------------------------------------------------------------
# TIME FILTER
# ---------------------------------------------------------------------------
st.sidebar.subheader('Time filter')
t_min = df_all['datetime'].min()
t_max = df_all['datetime'].max()

use_slider = st.sidebar.checkbox('Use interactive time-range slider', value=False)

if use_slider:
    # Use the first file as preview for slider
    preview_src = df_all['source'].iloc[0]
    df_prev = df_all[df_all['source'] == preview_src].copy()
    st.subheader(f'Time range preview – {preview_src}')
    fig_prev, ax_prev = plt.subplots(figsize=(12, 2))
    ax_prev.plot(df_prev['datetime'], df_prev['roll'], lw=0.5, label='Roll')
    ax_prev.set_ylabel('Roll (deg)')
    st.pyplot(fig_prev)
    plt.close(fig_prev)

    sel_start = st.sidebar.time_input('Start time', value=t_min.time())
    sel_end   = st.sidebar.time_input('End time',   value=t_max.time())
    start_dt  = datetime.combine(t_min.date(), sel_start)
    end_dt    = datetime.combine(t_max.date(), sel_end)
else:
    start_str = st.sidebar.text_input('Start (dd.mm.yyyy HH:MM:SS)', value=t_min.strftime('%d.%m.%Y %H:%M:%S'))
    end_str   = st.sidebar.text_input('End   (dd.mm.yyyy HH:MM:SS)', value=t_max.strftime('%d.%m.%Y %H:%M:%S'))
    try:
        start_dt = datetime.strptime(start_str, '%d.%m.%Y %H:%M:%S')
        end_dt   = datetime.strptime(end_str,   '%d.%m.%Y %H:%M:%S')
    except ValueError:
        st.error('Invalid date format – use dd.mm.yyyy HH:MM:SS')
        st.stop()

df_time = df_all[(df_all['datetime'] >= start_dt) & (df_all['datetime'] <= end_dt)].copy()

if df_time.empty:
    st.error('No data in selected time range.')
    st.stop()

# ---------------------------------------------------------------------------
# HEADING FILTER
# ---------------------------------------------------------------------------
st.sidebar.subheader('Heading filter')
hdg_mode = st.sidebar.radio('Mode', ['All headings', 'Target heading ± tolerance', 'Heading range'])

if hdg_mode == 'Target heading ± tolerance':
    center = st.sidebar.number_input('Target heading (deg)', 0.0, 360.0, 0.0)
    tol    = st.sidebar.number_input('Tolerance (deg)',       0.0, 180.0, 10.0)
    mask   = heading_in_range(df_time['heading'], center, tol)
    df_filt = df_time[mask].copy()
elif hdg_mode == 'Heading range':
    h_lo = st.sidebar.number_input('From (deg)', 0.0, 360.0, 0.0)
    h_hi = st.sidebar.number_input('To   (deg)', 0.0, 360.0, 360.0)
    if h_lo <= h_hi:
        df_filt = df_time[(df_time['heading'] >= h_lo) & (df_time['heading'] <= h_hi)].copy()
    else:  # wrap-around, e.g. 350 to 20
        df_filt = df_time[(df_time['heading'] >= h_lo) | (df_time['heading'] <= h_hi)].copy()
else:
    df_filt = df_time.copy()

if df_filt.empty:
    st.error('No data after heading filter.')
    st.stop()

# ---------------------------------------------------------------------------
# ROLLING RMS
# ---------------------------------------------------------------------------
for win, label in [(20, '20s'), (60, '60s'), (300, '300s')]:
    for col in ['roll', 'pitch', 'heave_m']:
        df_filt[f'rms_{col}_{label}'] = (
            df_filt[col].rolling(win, min_periods=1).apply(lambda x: float(np.sqrt((x**2).mean())))
        )

# ---------------------------------------------------------------------------
# OVERALL RMS TABLE
# ---------------------------------------------------------------------------
st.subheader('Overall RMS')
rms_rows = []
for src, grp in df_filt.groupby('source'):
    rms_rows.append({
        'File': src,
        'N rows': len(grp),
        'Roll RMS (deg)':  round(rms(grp['roll']),   4),
        'Pitch RMS (deg)': round(rms(grp['pitch']),  4),
        'Heave RMS (m)':   round(rms(grp['heave_m']),4),
    })
st.dataframe(pd.DataFrame(rms_rows), use_container_width=True)

# ---------------------------------------------------------------------------
# HEADING-BIN TABLE
# ---------------------------------------------------------------------------
st.subheader('Heading-bin RMS table')
bin_size = st.selectbox('Bin size (deg)', [5, 10, 15, 30, 45, 90], index=2)
bins = range(0, 360, bin_size)
bin_rows = []
for b in bins:
    mask = (df_filt['heading'] >= b) & (df_filt['heading'] < b + bin_size)
    grp  = df_filt[mask]
    bin_rows.append({
        'Heading bin': f'{b}-{b+bin_size}',
        'N':           len(grp),
        'Roll RMS':    round(rms(grp['roll']),   4) if len(grp) else np.nan,
        'Pitch RMS':   round(rms(grp['pitch']),  4) if len(grp) else np.nan,
        'Heave RMS':   round(rms(grp['heave_m']),4) if len(grp) else np.nan,
    })
df_bins = pd.DataFrame(bin_rows)
st.dataframe(df_bins, use_container_width=True)

# ---------------------------------------------------------------------------
# BAR CHARTS – RMS by heading bin
# ---------------------------------------------------------------------------
st.subheader('RMS by heading bin')
cols_plot = st.columns(3)
for ci, (col, label) in enumerate([('Roll RMS','Roll (deg)'),('Pitch RMS','Pitch (deg)'),('Heave RMS','Heave (m)')]):
    fig, ax = plt.subplots(figsize=(6, 3))
    df_plot = df_bins[df_bins['N'] > 0]
    ax.bar(df_plot['Heading bin'], df_plot[col])
    ax.set_xlabel('Heading bin (deg)')
    ax.set_ylabel(label)
    ax.set_title(col)
    plt.xticks(rotation=90, fontsize=7)
    plt.tight_layout()
    cols_plot[ci].pyplot(fig)
    plt.close(fig)

# ---------------------------------------------------------------------------
# POLAR CHARTS
# ---------------------------------------------------------------------------
st.subheader('Polar RMS charts')
cols_pol = st.columns(3)
for ci, (col, label) in enumerate([('Roll RMS','Roll'), ('Pitch RMS','Pitch'), ('Heave RMS','Heave')]):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(4, 4))
    df_plot = df_bins[df_bins['N'] > 0].copy()
    angles  = np.deg2rad([(b + bin_size/2) for b in range(0, 360, bin_size) if df_bins.iloc[b//bin_size]['N'] > 0])
    values  = df_plot[col].values
    if len(angles) and len(values):
        ax.plot(np.append(angles, angles[0]), np.append(values, values[0]))
        ax.fill(np.append(angles, angles[0]), np.append(values, values[0]), alpha=0.25)
    ax.set_title(label, va='bottom')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    cols_pol[ci].pyplot(fig)
    plt.close(fig)

# ---------------------------------------------------------------------------
# EXPORT
# ---------------------------------------------------------------------------
st.subheader('Export')
export_zip = io.BytesIO()
with zipfile.ZipFile(export_zip, 'w') as zf:
    # Per-file CSVs
    for src, grp in df_filt.groupby('source'):
        csv_bytes = grp.to_csv(index=False).encode()
        zf.writestr(src.replace('.NPD','').replace('.npd','') + '_filtered.csv', csv_bytes)
    # Bin table
    zf.writestr('heading_bin_rms.csv', df_bins.to_csv(index=False).encode())
export_zip.seek(0)
st.download_button('Download ZIP', data=export_zip, file_name='navipac_rms_export.zip', mime='application/zip')
