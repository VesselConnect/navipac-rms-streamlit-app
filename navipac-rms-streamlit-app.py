def heave_acceleration_rms(series: pd.Series, dt_seconds: float, low_hz=None, high_hz=None) -> float:
    s = pd.to_numeric(series, errors="coerce")
    if len(s.dropna()) < 3 or dt_seconds <= 0:
        return float("nan")

    vals = s.to_numpy(dtype=float)
    if not np.isfinite(vals).all():
        vals = pd.Series(vals).interpolate(limit_direction="both").to_numpy(dtype=float)

    vals = vals - np.mean(vals)
    fs = 1.0 / dt_seconds

    freqs = np.fft.rfftfreq(len(vals), d=dt_seconds)
    spec = np.fft.rfft(vals)
    mask = np.ones_like(freqs, dtype=bool)

    if low_hz is not None and low_hz > 0:
        mask &= freqs >= low_hz
    if high_hz is not None and high_hz > 0:
        mask &= freqs <= high_hz

    spec[~mask] = 0
    filtered = np.fft.irfft(spec, n=len(vals))

    acc = np.gradient(np.gradient(filtered, dt_seconds), dt_seconds)
    return float(np.sqrt(np.mean(acc ** 2))) if len(acc) else float("nan")
