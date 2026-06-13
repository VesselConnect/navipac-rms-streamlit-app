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
# PARSER
# ---------------------------------------------------------------------------
LINE_RE = re.compile(
    r'^(\d{2}\.\d{2}\.\d{4})\s+(\d{2}:\d{2}:\d{2})'
    r'\s+([\d.eE+\-]+)'   # heading
    r'\s+([\d.eE+\-]+)'   # roll
    r'\s+([\d.eE+\-]+)'   # pitch
    r'\s+([\d.eE+\-]+)'   # heave
)


def parse_npd(file_bytes: bytes, filename: str) -> pd.DataFrame:
    text = file_bytes.decode('latin-1', errors='replace')
    rows = []
    for line in text.splitlines():
        m = LINE_RE.match(line.strip())
        if m:
            date_str, time_str, hdg, roll, pitch, heave = m.groups()
            dt = datetime.strptime(f'{date_str} {time_str}', '%d.%m.%Y %H:%M:%S')
            rows.append({
                'datetime': dt,
                'heading': float(hdg),
                'roll': float(roll),
                'pitch': float(pitch),
                'heave_m': float(heave),
            })
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
# SIDEBAR – file upload
# ---------------------------------------------------------------------------
st.sidebar.header('Upload NPD files')
uploaded = st.sidebar.file_uploader(
    'Select one or more .NPD files',
    type=['npd', 'NPD', 'txt'],
    accept_multiple_files=True,
)

if not uploaded:
    st.info('Upload one or more NaviPac .NPD files to begin.')
    st.stop()

# Parse all files
all_frames = []
for f in uploaded:
    df_f = parse_npd(f.read(), f.name)
    if not df_f.empty:
        all_frames.append(df_f)
    else:
        st.warning(f'No parseable rows found in {f.name}')

if not all_frames:
    st.error('No data could be parsed from the uploaded files.')
    st.stop()

all_data = pd.concat(all_frames, ignore_index=True).sort_values('datetime')

# ---------------------------------------------------------------------------
# SIDEBAR – time filter
# ---------------------------------------------------------------------------
st.sidebar.header('Time filter')
t_min = all_data['datetime'].min()
t_max = all_data['datetime'].max()

use_slider = st.sidebar.checkbox('Use interactive time-range slider', value=False)

if use_slider:
    # Use the first file for the preview plot
    preview_df = all_frames[0].copy()
    fig_prev, ax_prev = plt.subplots(figsize=(8, 2))
    ax_prev.plot(preview_df['datetime'], preview_df['heave_m'], lw=0.6)
    ax_prev.set_ylabel('Heave (m)')
    ax_prev.set_title(f'Preview: {all_frames[0]["source"].iloc[0]}')
    st.pyplot(fig_prev)
    plt.close(fig_prev)

    total_seconds = int((t_max - t_min).total_seconds())
    slider_range = st.slider(
        'Select time window (seconds from start)',
        0, total_seconds, (0, total_seconds),
    )
    t_start = t_min + pd.Timedelta(seconds=slider_range[0])
    t_end   = t_min + pd.Timedelta(seconds=slider_range[1])
else:
    date_range = st.sidebar.date_input(
        'Date range',
        value=[t_min.date(), t_max.date()],
        min_value=t_min.date(),
        max_value=t_max.date(),
    )
    if len(date_range) == 2:
        t_start = datetime.combine(date_range[0], time.min)
        t_end   = datetime.combine(date_range[1], time.max)
    else:
        t_start, t_end = t_min, t_max

# ---------------------------------------------------------------------------
# SIDEBAR – heading filter
# ---------------------------------------------------------------------------
st.sidebar.header('Heading filter')
hdg_mode = st.sidebar.radio(
    'Heading mode',
    ['All headings', 'Target heading ± tolerance', 'Heading range'],
)
hdg_mask = None
if hdg_mode == 'Target heading ± tolerance':
    center = st.sidebar.number_input('Target heading (°)', 0.0, 360.0, 0.0, 1.0)
    tol    = st.sidebar.number_input('Tolerance (°)', 0.0, 180.0, 15.0, 1.0)
elif hdg_mode == 'Heading range':
    h_lo = st.sidebar.number_input('From heading (°)', 0.0, 360.0, 0.0, 1.0)
    h_hi = st.sidebar.number_input('To heading (°)',   0.0, 360.0, 90.0, 1.0)

# ---------------------------------------------------------------------------
# SIDEBAR – bin size
# ---------------------------------------------------------------------------
st.sidebar.header('Heading bin size')
bin_size = st.sidebar.selectbox('Bin size (°)', [5, 10, 15, 30, 45, 90], index=2)

# ---------------------------------------------------------------------------
# APPLY FILTERS
# ---------------------------------------------------------------------------
mask_time = (
    (all_data['datetime'] >= t_start) &
    (all_data['datetime'] <= t_end)
)
df_filtered = all_data[mask_time].copy()

if hdg_mode == 'Target heading ± tolerance':
    df_filtered = df_filtered[heading_in_range(df_filtered['heading'], center, tol)]
elif hdg_mode == 'Heading range':
    if h_lo <= h_hi:
        df_filtered = df_filtered[
            (df_filtered['heading'] >= h_lo) & (df_filtered['heading'] <= h_hi)
        ]
    else:  # wraps around
        df_filtered = df_filtered[
            (df_filtered['heading'] >= h_lo) | (df_filtered['heading'] <= h_hi)
        ]

if df_filtered.empty:
    st.warning('No data in the selected window / heading filter.')
    st.stop()

# Rolling RMS
for win, label in [(20, '20s'), (60, '60s'), (300, '300s')]:
    for col in ['roll', 'pitch', 'heave_m']:
        df_filtered[f'{col}_rms_{label}'] = (
            df_filtered[col]
            .pow(2)
            .rolling(window=win, min_periods=1)
            .mean()
            .pow(0.5)
        )

# ---------------------------------------------------------------------------
# OVERALL RMS
# ---------------------------------------------------------------------------
st.header('Overall RMS (filtered data)')
col1, col2, col3 = st.columns(3)
col1.metric('Roll RMS (°)',   f"{rms(df_filtered['roll']):.3f}")
col2.metric('Pitch RMS (°)',  f"{rms(df_filtered['pitch']):.3f}")
col3.metric('Heave RMS (m)', f"{rms(df_filtered['heave_m']):.3f}")

# ---------------------------------------------------------------------------
# HEADING BIN TABLE
# ---------------------------------------------------------------------------
st.header(f'RMS by heading bin ({bin_size}°)')
bins = np.arange(0, 360 + bin_size, bin_size)
labels = [f'{int(b)}-{int(b+bin_size)}°' for b in bins[:-1]]
df_filtered['hdg_bin'] = pd.cut(
    df_filtered['heading'], bins=bins, labels=labels, right=False, include_lowest=True
)
bin_table = (
    df_filtered.groupby('hdg_bin', observed=False)[['roll', 'pitch', 'heave_m']]
    .apply(lambda g: pd.Series({
        'Roll RMS (°)':  rms(g['roll']),
        'Pitch RMS (°)': rms(g['pitch']),
        'Heave RMS (m)': rms(g['heave_m']),
        'N samples':     len(g),
    }))
    .reset_index()
)
st.dataframe(bin_table, use_container_width=True)

# ---------------------------------------------------------------------------
# BAR CHARTS – RMS by heading bin
# ---------------------------------------------------------------------------
st.header('RMS by heading bin – bar charts')
fig_bar, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
for ax, col, title in zip(
    axes,
    ['Roll RMS (°)', 'Pitch RMS (°)', 'Heave RMS (m)'],
    ['Roll RMS', 'Pitch RMS', 'Heave RMS'],
):
    valid = bin_table.dropna(subset=[col])
    ax.bar(valid['hdg_bin'].astype(str), valid[col], color='steelblue')
    ax.set_title(title)
    ax.set_xlabel('Heading bin')
    ax.set_ylabel(col)
    ax.tick_params(axis='x', rotation=90)
plt.tight_layout()
st.pyplot(fig_bar)
plt.close(fig_bar)

# ---------------------------------------------------------------------------
# POLAR CHARTS
# ---------------------------------------------------------------------------
st.header('Polar RMS charts')
fig_pol, pol_axes = plt.subplots(1, 3, figsize=(15, 5),
                                   subplot_kw={'projection': 'polar'})
theta = np.deg2rad(
    [(b + bin_size / 2) for b in bins[:-1]]
)
for ax, col, title in zip(
    pol_axes,
    ['Roll RMS (°)', 'Pitch RMS (°)', 'Heave RMS (m)'],
    ['Roll RMS', 'Pitch RMS', 'Heave RMS'],
):
    r_vals = bin_table[col].fillna(0).values
    ax.bar(theta, r_vals, width=np.deg2rad(bin_size), bottom=0,
           align='center', alpha=0.75, color='teal')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_title(title, pad=15)
plt.tight_layout()
st.pyplot(fig_pol)
plt.close(fig_pol)

# ---------------------------------------------------------------------------
# ROLLING RMS TIME SERIES
# ---------------------------------------------------------------------------
st.header('Rolling RMS time series')
for col, unit in [('roll', '°'), ('pitch', '°'), ('heave_m', 'm')]:
    fig_ts, ax_ts = plt.subplots(figsize=(12, 3))
    ax_ts.plot(df_filtered['datetime'], df_filtered[f'{col}_rms_20s'],  label='20 s', lw=0.8)
    ax_ts.plot(df_filtered['datetime'], df_filtered[f'{col}_rms_60s'],  label='60 s', lw=0.8)
    ax_ts.plot(df_filtered['datetime'], df_filtered[f'{col}_rms_300s'], label='300 s', lw=0.8)
    ax_ts.set_title(f'{col.capitalize()} rolling RMS')
    ax_ts.set_ylabel(f'RMS ({unit})')
    ax_ts.legend()
    plt.tight_layout()
    st.pyplot(fig_ts)
    plt.close(fig_ts)

# ---------------------------------------------------------------------------
# EXPORT
# ---------------------------------------------------------------------------
st.header('Export')

# Per-file CSV
zip_buf = io.BytesIO()
with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
    for src_name, grp in df_filtered.groupby('source'):
        csv_bytes = grp.to_csv(index=False).encode()
        safe_name = src_name.replace('/', '_').replace('\\', '_')
        zf.writestr(f'{safe_name}_rms.csv', csv_bytes)
    # Bin table
    zf.writestr('heading_bin_table.csv', bin_table.to_csv(index=False).encode())
zip_buf.seek(0)

st.download_button(
    label='Download ZIP (CSV + bin table)',
    data=zip_buf,
    file_name='navipac_rms_export.zip',
    mime='application/zip',
)
