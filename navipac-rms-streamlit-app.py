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
                "heading_deg": 
