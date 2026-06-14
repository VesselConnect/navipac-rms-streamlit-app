import io
import re
import zipfile
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="NaviPac NPD RMS", layout="wide")


@dataclass
class FilterConfig:
    mode: str
    target_heading: float | None = None
    tolerance: float | None = None
    min_heading: float | None = None
    max_heading: float | None = None


def parse_float_safe(value: str):
    if value is None:
        return None
    s = str(value).strip()
    if s == "":
        return None
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def parse_timestamp_safe(text: str):
    text = str(text).strip()
    formats = [
        "%d.%m.%Y %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%d.%m.%y %H:%M:%S",
    ]
    for fmt in formats:
        ts = pd.to_datetime(text, format=fmt, errors="coerce")
        if pd.notna(ts):
            return ts
    return pd.NaT


def parse_npd_text(text: str, source_name: str = "uploaded_file") -> pd.DataFrame:
    rows = []
    skipped = 0

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        lower = line.lower()
        if any(tag in lower for tag in ["time", "date", "heading", "roll", "pitch", "heave"]):
            continue

        line = re.sub(r"[;\t]+", ",", line)
        parts = [p.strip() for p in line.split(",") if p.strip() != ""]

        if len(parts) < 5:
            skipped += 1
            continue

        ts = pd.NaT
        data_start_idx = None

        if len(parts) >= 2:
            ts = parse_timestamp_safe(f"{parts[0]} {parts[1]}")
            if pd.notna(ts):
                data_start_idx = 2

        if pd.isna(ts):
            ts = parse_timestamp_safe(parts[0])
            if pd.notna(ts):
                data_start_idx = 1

        if pd.isna(ts) or data_start_idx is None:
            skipped += 1
            continue

        numeric_vals = []
        for p in parts[data_start_idx:]:
            val = parse_float_safe(p)
            if val is not None:
                numeric_vals.append(val)

        if len(numeric_vals) < 4:
            skipped += 1
            continue

        heading, roll, pitch, heave = numeric_vals[:4]

        rows.append(
            {
                "timestamp": ts,
                "heading_deg": heading,
                "roll_deg": roll,
                "pitch_deg": pitch,
                "heave_m": heave,
                "source_file": source_name,
                "raw_line": raw_line,
            }
        )

    if not rows:
        sample = text[:1000].replace("\n", " ")
        raise ValueError(
            f"No valid motion rows found in {source_name}. "
            f"Parser expected a timestamp plus at least 4 numeric motion fields. "
            f"First 1000 chars: {sample}"
        )

    df = (
        pd.DataFrame(rows)
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"])
        .reset_index(drop=True)
    )

    return df


def parse_npd_file(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.read()
    text = raw.decode("utf-8", errors="ignore") if isinstance(raw, (bytes, bytearray)) else str(raw)
    return parse_npd_text(text, source_name=getattr(uploaded_file, "name", "uploaded_file"))


def estimate_dt_seconds(df: pd.DataFrame) -> float:
    if len(df) < 2:
        return 1.0
    dt = df["timestamp"].diff().dt.total_seconds().dropna()
    dt = dt[dt > 0]
    if dt.empty:
        return 1.0
    return float(dt.median())


def rms(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(s))))


def detrend_linear(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if len(s.dropna()) < 3:
        return s - s.mean()
    idx = np.arange(len(s), dtype=float)
    vals = s.to_numpy(dtype=float)
    mask = np.isfinite(vals)
    if mask.sum() < 3:
        return pd.Series(vals - np.nanmean(vals), index=s.index)
    coeffs = np.polyfit(idx[mask], vals[mask], 1)
    trend = coeffs[0] * idx + coeffs[1]
    return pd.Series(vals - trend, index=s.index)


def fft_bandpass_series(series: pd.Series, dt_seconds: float, low_hz=None, high_hz=None) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    vals = s.to_numpy(dtype=float)
    if len(vals) < 4 or dt_seconds <= 0:
        return s - s.mean()

    x = vals.copy()
    mask_valid = np.isfinite(x)
    if mask_valid.sum() < 4:
        return pd.Series(x - np.nanmean(x), index=s.index)

    if not mask_valid.all():
        x = pd.Series(x).interpolate(limit_direction="both").to_numpy(dtype=float)

    x = x - np.mean(x)
    freqs = np.fft.rfftfreq(len(x), d=dt_seconds)
    spec = np.fft.rfft(x)

    mask = np.ones_like(freqs, dtype=bool)
    if low_hz is not None and low_hz > 0:
        mask &= freqs >= low_hz
    if high_hz is not None and high_hz > 0:
        mask &= freqs <= high_hz

    spec[~mask] = 0
    out = np.fft.irfft(spec, n=len(x))
    return pd.Series(out, index=s.index)


def max_heave_amplitude(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return float("nan")
    centered = s - s.mean()
    return float(np.max(np.abs(centered)))


def heave_acceleration_rms(series: pd.Series, dt_seconds: float, low_hz=None, high_hz=None) -> float:
    filtered = fft_bandpass_series(series, dt_seconds, low_hz=low_hz, high_hz=high_hz)
    vals = filtered.to_numpy(dtype=float)
    if len(vals) < 3 or dt_seconds <= 0:
        return float("nan")
    acc = np.gradient(np.gradient(vals, dt_seconds), dt_seconds)
    return float(np.sqrt(np.mean(acc ** 2))) if len(acc) else float("nan")


def motion_diagnostics(df: pd.DataFrame, low_hz: float = 0.03, high_hz: float = 0.5) -> pd.DataFrame:
    dt = estimate_dt_seconds(df)
    rows = []
    mapping = [("roll_deg", "deg"), ("pitch_deg", "deg"), ("heave_m", "m")]

    for col, unit in mapping:
        raw = pd.to_numeric(df[col], errors="coerce")
        demeaned = raw - raw.mean()
        detrended = detrend_linear(raw)
        filtered = fft_bandpass_series(detrended, dt, low_hz=low_hz, high_hz=high_hz)

        rows.append(
            {
                "metric": col,
                "unit": unit,
                "mean_raw": float(raw.mean()),
                "rms_raw": rms(raw),
                "rms_demeaned": rms(demeaned),
                "rms_detrended": rms(detrended),
                "rms_filtered": rms(filtered),
                "sample_dt_s": dt,
                "max_heave_amplitude": max_heave_amplitude(raw) if col == "heave_m" else np.nan,
                "heave_accel_rms_m_s2": heave_acceleration_rms(
                    detrended, dt, low_hz=low_hz, high_hz=high_hz
                ) if col == "heave_m" else np.nan,
            }
        )

    return pd.DataFrame(rows)


def rolling_rms(series: pd.Series, window_seconds: int, dt_seconds: float) -> pd.Series:
    window_samples = max(1, int(round(window_seconds / max(dt_seconds, 1e-9))))
    vals = pd.to_numeric(series, errors="coerce")
    return np.sqrt(vals.pow(2).rolling(window_samples, min_periods=max(2, window_samples // 5)).mean())


def heading_in_band(headings: pd.Series, min_h: float, max_h: float) -> pd.Series:
    h = headings % 360
    min_h = min_h % 360
    max_h = max_h % 360
    if min_h <= max_h:
        return (h >= min_h) & (h <= max_h)
    return (h >= min_h) | (h <= max_h)


def apply_heading_filter(df: pd.DataFrame, cfg: FilterConfig) -> pd.DataFrame:
    if cfg.mode == "All headings":
        return df.copy()
    if cfg.mode == "Target heading ± tolerance":
        target = float(cfg.target_heading or 0.0) % 360
        tol = abs(float(cfg.tolerance or 0.0))
        delta = ((df["heading_deg"] - target + 180) % 360) - 180
        return df[delta.abs() <= tol].copy()
    if cfg.mode == "Heading range":
        return df[
            heading_in_band(
                df["heading_deg"],
                float(cfg.min_heading or 0.0),
                float(cfg.max_heading or 0.0),
            )
        ].copy()
    return df.copy()


def enrich_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = estimate_dt_seconds(out)
    for sec in (20, 60, 300):
        out[f"roll_rms_{sec}s"] = rolling_rms(out["roll_deg"], sec, dt)
        out[f"pitch_rms_{sec}s"] = rolling_rms(out["pitch_deg"], sec, dt)
        out[f"heave_rms_{sec}s"] = rolling_rms(out["heave_m"], sec, dt)
    return out


def build_summary(file_name: str, raw_df: pd.DataFrame, filt_df: pd.DataFrame) -> dict:
    dt = estimate_dt_seconds(raw_df)
    duration_s = (
        max(0.0, (raw_df["timestamp"].max() - raw_df["timestamp"].min()).total_seconds())
        if len(raw_df)
        else 0.0
    )

    return {
        "file_name": file_name,
        "rows_total": int(len(raw_df)),
        "rows_filtered": int(len(filt_df)),
        "start_time": raw_df["timestamp"].min(),
        "end_time": raw_df["timestamp"].max(),
        "duration_s": duration_s,
        "sample_dt_s": dt,
        "sample_rate_hz": (1.0 / dt) if dt > 0 else np.nan,
        "heading_mean_deg": float(filt_df["heading_deg"].mean()) if len(filt_df) else np.nan,
        "roll_rms_deg": rms(filt_df["roll_deg"]),
        "pitch_rms_deg": rms(filt_df["pitch_deg"]),
        "heave_rms_m": rms(filt_df["heave_m"]),
        "roll_max_abs_deg": float(filt_df["roll_deg"].abs().max()) if len(filt_df) else np.nan,
        "pitch_max_abs_deg": float(filt_df["pitch_deg"].abs().max()) if len(filt_df) else np.nan,
        "heave_max_abs_m": float(filt_df["heave_m"].abs().max()) if len(filt_df) else np.nan,
    }


def make_plot(df: pd.DataFrame, file_name: str):
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(df["timestamp"], df["heading_deg"], lw=1)
    axes[0].set_ylabel("Heading")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["timestamp"], df["roll_deg"], lw=1, label="Roll")
    if "roll_rms_60s" in df:
        axes[1].plot(df["timestamp"], df["roll_rms_60s"], lw=1.2, label="60 s RMS")
    axes[1].set_ylabel("Roll")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(df["timestamp"], df["pitch_deg"], lw=1, label="Pitch")
    if "pitch_rms_60s" in df:
        axes[2].plot(df["timestamp"], df["pitch_rms_60s"], lw=1.2, label="60 s RMS")
    axes[2].set_ylabel("Pitch")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(df["timestamp"], df["heave_m"], lw=1, label="Heave")
    if "heave_rms_60s" in df:
        axes[3].plot(df["timestamp"], df["heave_rms_60s"], lw=1.2, label="60 s RMS")
    axes[3].set_ylabel("Heave")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    axes[3].set_xlabel("Time")

    fig.suptitle(file_name)
    fig.tight_layout()
    return fig


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def build_zip(results: dict[str, dict]) -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        summaries = []
        diagnostics_tables = []

        for name, payload in results.items():
            summaries.append(payload["summary"])
            zf.writestr(f"parsed/{name}.parsed.csv", payload["raw_df"].to_csv(index=False))
            zf.writestr(f"filtered/{name}.filtered.csv", payload["filtered_df"].to_csv(index=False))

            if "diagnostics_df" in payload and payload["diagnostics_df"] is not None:
                diag = payload["diagnostics_df"].copy()
                diag.insert(0, "file_name", name)
                diagnostics_tables.append(diag)
                zf.writestr(f"diagnostics/{name}.diagnostics.csv", diag.to_csv(index=False))

            fig = make_plot(payload["filtered_df"], name)
            img = io.BytesIO()
            fig.savefig(img, format="png", dpi=160, bbox_inches="tight")
            plt.close(fig)
            img.seek(0)
            zf.writestr(f"plots/{name}.png", img.read())

        zf.writestr("summary/rms_summary.csv", pd.DataFrame(summaries).to_csv(index=False))
        if diagnostics_tables:
            zf.writestr(
                "summary/motion_diagnostics.csv",
                pd.concat(diagnostics_tables, ignore_index=True).to_csv(index=False),
            )

    mem.seek(0)
    return mem.read()


st.title("NaviPac NPD RMS extractor")
st.write("Upload NaviPac `.NPD` files, filter by heading, and inspect RMS, heave amplitude, and heave acceleration.")

with st.sidebar:
    st.header("Heading filter")
    mode = st.selectbox("Mode", ["All headings", "Target heading ± tolerance", "Heading range"])

    target_heading = tolerance = min_heading = max_heading = None
    if mode == "Target heading ± tolerance":
        target_heading = st.number_input("Target heading (deg)", value=170.0, min_value=0.0, max_value=360.0, step=0.1)
        tolerance = st.number_input("Tolerance (deg)", value=5.0, min_value=0.0, max_value=180.0, step=0.1)
    elif mode == "Heading range":
        min_heading = st.number_input("Min heading (deg)", value=165.0, min_value=0.0, max_value=360.0, step=0.1)
        max_heading = st.number_input("Max heading (deg)", value=175.0, min_value=0.0, max_value=360.0, step=0.1)

cfg = FilterConfig(
    mode=mode,
    target_heading=target_heading,
    tolerance=tolerance,
    min_heading=min_heading,
    max_heading=max_heading,
)

uploaded_files = st.file_uploader("Upload NPD file(s)", type=["npd", "txt"], accept_multiple_files=True)

if uploaded_files:
    results = {}
    parse_errors = {}

    diag_low_hz = st.number_input("Diagnostics low-cut frequency (Hz)", min_value=0.0, value=0.03, step=0.01, format="%.2f")
    diag_high_hz = st.number_input("Diagnostics high-cut frequency (Hz)", min_value=0.0, value=0.50, step=0.01, format="%.2f")

    for f in uploaded_files:
        try:
            raw_df = parse_npd_file(f)
            filtered_df = apply_heading_filter(raw_df, cfg)
            filtered_df = enrich_df(filtered_df) if not filtered_df.empty else filtered_df.copy()
            diagnostics_df = (
                motion_diagnostics(
                    filtered_df,
                    low_hz=diag_low_hz if diag_low_hz > 0 else None,
                    high_hz=diag_high_hz if diag_high_hz > 0 else None,
                )
                if not filtered_df.empty
                else None
            )

            results[f.name] = {
                "raw_df": raw_df,
                "filtered_df": filtered_df,
                "diagnostics_df": diagnostics_df,
                "summary": build_summary(f.name, raw_df, filtered_df),
            }
        except Exception as e:
            parse_errors[f.name] = str(e)

    if parse_errors:
        st.error("Some files failed to parse:")
        for name, err in parse_errors.items():
            st.write(f"- {name}: {err}")

    if results:
        summary_df = pd.DataFrame([v["summary"] for v in results.values()])
        st.subheader("RMS summary")
        st.dataframe(summary_df, use_container_width=True)

        st.download_button(
            "Download RMS summary CSV",
            data=to_csv_bytes(summary_df),
            file_name="rms_summary.csv",
            mime="text/csv",
        )

        zip_bytes = build_zip(results)
        st.download_button(
            "Download ZIP package",
            data=zip_bytes,
            file_name="navipac_npd_rms_outputs.zip",
            mime="application/zip",
        )

        selected = st.selectbox("Inspect file", list(results.keys()))
        payload = results[selected]

        meta1, meta2, meta3, meta4 = st.columns(4)
        meta1.metric("Parsed rows", len(payload["raw_df"]))
        meta2.metric("Filtered rows", len(payload["filtered_df"]))
        meta3.metric("Start", str(payload["raw_df"]["timestamp"].min()))
        meta4.metric("End", str(payload["raw_df"]["timestamp"].max()))

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Parsed preview")
            st.dataframe(payload["raw_df"].head(20), use_container_width=True)
        with c2:
            st.markdown("### Filtered preview")
            st.dataframe(payload["filtered_df"].head(20), use_container_width=True)

        if payload["diagnostics_df"] is not None:
            st.subheader("Motion diagnostics")
            st.dataframe(payload["diagnostics_df"], use_container_width=True)

            heave_diag = payload["diagnostics_df"][payload["diagnostics_df"]["metric"] == "heave_m"].copy()
            if not heave_diag.empty:
                st.subheader("Heave extra outputs")
                h1, h2 = st.columns(2)
                h1.metric("Maximum heave amplitude", f"{heave_diag['max_heave_amplitude'].iloc[0]:.3f} m")
                h2.metric("RMS heave acceleration", f"{heave_diag['heave_accel_rms_m_s2'].iloc[0]:.3f} m/s²")

            st.caption(
                "Raw RMS includes static offset or bias. Demeaned RMS removes average offset. "
                "Detrended RMS also removes linear drift. Filtered RMS applies a simple FFT band-pass "
                "to the detrended signal. Maximum heave amplitude is the peak absolute heave about the mean. "
                "RMS heave acceleration is derived from the band-passed detrended heave signal using a second numerical derivative."
            )

        st.markdown("### Plot")
        if payload["filtered_df"].empty:
            st.warning("No rows matched the selected heading filter.")
        else:
            fig = make_plot(payload["filtered_df"], selected)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
else:
    st.info("Upload one or more NaviPac NPD files to begin.")
