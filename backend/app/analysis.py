from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .models import AnalysisOptions
from .series import RainfallSeries


def _clean_number(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _clean_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_value(v) for v in value]
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return _clean_number(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def _mann_kendall(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    n = len(y)
    if n < 3:
        return {"s": None, "z": None, "p": None, "tau": None, "significant_5pct": None}
    delta = y[np.newaxis, :] - y[:, np.newaxis]
    s = int(np.sign(delta[np.triu_indices(n, 1)]).sum())
    _, counts = np.unique(y, return_counts=True)
    tie_term = sum(int(t) * (int(t) - 1) * (2 * int(t) + 5) for t in counts if t > 1)
    variance = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    if variance <= 0:
        z = 0.0
        p = 1.0
    elif s > 0:
        z = (s - 1.0) / math.sqrt(variance)
        p = float(2 * stats.norm.sf(abs(z)))
    elif s < 0:
        z = (s + 1.0) / math.sqrt(variance)
        p = float(2 * stats.norm.sf(abs(z)))
    else:
        z, p = 0.0, 1.0
    n0 = n * (n - 1) / 2
    tie_pairs = sum(int(t) * (int(t) - 1) / 2 for t in counts if t > 1)
    tau = s / math.sqrt(n0 * (n0 - tie_pairs)) if n0 > tie_pairs else 0.0
    return {"s": s, "z": z, "p": p, "tau": tau, "significant_5pct": p < 0.05}


def _sen_slope(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(y) < 2:
        return None
    return _clean_number(stats.theilslopes(y, x, alpha=0.95).slope)


def _pettitt(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    n = len(y)
    if n < 3:
        return {"change_year": None, "statistic": None, "p": None}
    ranks = stats.rankdata(y, method="average")
    u = 2 * np.cumsum(ranks) - np.arange(1, n + 1) * (n + 1)
    u[-1] = 0
    index = int(np.argmax(np.abs(u)))
    k = float(abs(u[index]))
    p = min(1.0, 2.0 * math.exp((-6.0 * k * k) / (n**3 + n**2)))
    return {"change_year": int(x[index]), "statistic": k, "p": p}


def trend_summary(years: list[int], values: list[float | None]) -> dict[str, Any]:
    pairs = [(int(x), float(y)) for x, y in zip(years, values) if y is not None and math.isfinite(float(y))]
    if len(pairs) < 3:
        return {"n": len(pairs), "message": "At least 3 valid annual values are required for trend tests."}
    x = np.asarray([p[0] for p in pairs], dtype=float)
    y = np.asarray([p[1] for p in pairs], dtype=float)
    mk = _mann_kendall(x, y)
    sen = _sen_slope(x, y)
    spearman = stats.spearmanr(x, y)
    regression = stats.linregress(x, y)

    # Trend-free pre-whitening: estimate and remove Sen trend, pre-whiten
    # residuals by lag-1 correlation, then restore the estimated trend.
    slope = sen or 0.0
    residual = y - slope * (x - x[0])
    r1 = float(np.corrcoef(residual[1:], residual[:-1])[0, 1]) if len(y) > 3 and np.std(residual) > 0 else 0.0
    r1 = min(0.99, max(-0.99, r1 if math.isfinite(r1) else 0.0))
    prewhitened = residual.copy()
    prewhitened[1:] = residual[1:] - r1 * residual[:-1]
    tfpw_series = prewhitened + slope * (x - x[0])
    tfpw = _mann_kendall(x, tfpw_series)
    return _clean_value({
        "n": len(y), "mk": mk, "sen_slope_per_year": sen,
        "spearman": {"rho": spearman.statistic, "p": spearman.pvalue},
        "linear_regression": {"slope_per_year": regression.slope, "intercept": regression.intercept,
                              "r_squared": regression.rvalue**2, "p": regression.pvalue},
        "pettitt": _pettitt(x, y),
        "tfpw": {"mk": tfpw, "lag1_autocorrelation": r1,
                 "sen_slope_per_year": slope, "method_note": "Trend-free pre-whitening using Sen slope and lag-1 Pearson correlation."},
    })


def _aggregate_monthly(series: RainfallSeries) -> pd.Series:
    values = series.values
    if series.frequency == "daily":
        return values.resample("MS").sum(min_count=1)
    if series.frequency == "monthly":
        return values.resample("MS").sum(min_count=1)
    return pd.Series(dtype=float)


def _annual_series(series: RainfallSeries, options: AnalysisOptions) -> tuple[pd.Series, pd.Series]:
    if series.frequency == "annual":
        raw = series.values.copy()
    elif series.frequency in {"daily", "monthly"}:
        raw = series.values.resample("YS").sum(min_count=1)
    else:
        raw = series.values.resample("YS").sum(min_count=1)
    years = raw.index.year
    if series.frequency == "daily":
        counts = series.values.resample("YS").count()
        expected = pd.Series([366 if t.is_leap_year else 365 for t in counts.index], index=counts.index)
        coverage = counts / expected * 100
        raw = raw.where(coverage >= (100 - options.max_missing_per_year_pct))
    elif series.frequency == "monthly":
        counts = series.values.resample("YS").count()
        raw = raw.where(counts >= 12 * (1 - options.max_missing_per_year_pct / 100))
    return pd.Series(raw.to_numpy(dtype=float), index=years), raw


def _longest_run(mask: np.ndarray) -> int:
    best = current = 0
    for item in mask:
        current = current + 1 if bool(item) else 0
        best = max(best, current)
    return best


def _extremes_daily(series: RainfallSeries, options: AnalysisOptions) -> list[dict[str, Any]] | None:
    if series.frequency != "daily":
        return None
    daily = series.values.asfreq("D")
    wet = daily[daily >= options.rainy_day_mm]
    base = daily
    if options.base_period_start:
        base = base[base.index.year >= options.base_period_start]
    if options.base_period_end:
        base = base[base.index.year <= options.base_period_end]
    wet_base = base[base >= options.rainy_day_mm]
    q95 = float(wet_base.quantile(0.95)) if not wet_base.empty else None
    q99 = float(wet_base.quantile(0.99)) if not wet_base.empty else None
    rows = []
    for year, values in daily.groupby(daily.index.year):
        valid = values.dropna()
        if valid.empty:
            continue
        rolling5 = values.rolling(5, min_periods=5).sum()
        valid_wet = valid[valid >= options.rainy_day_mm]
        rows.append({
            "year": int(year), "PRCPTOT_mm": float(valid_wet.sum()),
            "Rx1day_mm": float(valid.max()), "Rx5day_mm": _clean_number(rolling5.max()),
            "R10mm_days": int((valid >= 10).sum()), "R20mm_days": int((valid >= 20).sum()),
            "SDII_mm_per_wet_day": _clean_number(valid_wet.mean()),
            "CDD_days": _longest_run((values.notna() & (values < options.rainy_day_mm)).to_numpy()),
            "CWD_days": _longest_run((values.notna() & (values >= options.rainy_day_mm)).to_numpy()),
            "R95p_mm": float(valid[valid > q95].sum()) if q95 is not None else None,
            "R99p_mm": float(valid[valid > q99].sum()) if q99 is not None else None,
        })
    return _clean_value(rows)


def _spi(monthly: pd.Series, options: AnalysisOptions) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for scale in options.spi_scales:
        accumulated = monthly.rolling(scale, min_periods=scale).sum()
        spi = pd.Series(np.nan, index=accumulated.index, dtype=float)
        for month in range(1, 13):
            mask = accumulated.index.month == month
            sample = accumulated.loc[mask].dropna()
            if options.base_period_start:
                sample = sample[sample.index.year >= options.base_period_start]
            if options.base_period_end:
                sample = sample[sample.index.year <= options.base_period_end]
            positive = sample[sample > 0].to_numpy(dtype=float)
            if len(positive) < 10:
                continue
            try:
                shape, _, scale_param = stats.gamma.fit(positive, floc=0)
                zero_probability = float((sample <= 0).mean())
                all_values = accumulated.loc[mask]
                probability = zero_probability + (1 - zero_probability) * stats.gamma.cdf(
                    np.maximum(all_values.to_numpy(dtype=float), 0), a=shape, loc=0, scale=scale_param
                )
                probability = np.clip(probability, 1e-6, 1 - 1e-6)
                spi.loc[mask] = stats.norm.ppf(probability)
            except (ValueError, FloatingPointError):
                continue
        events = _drought_events(spi, options.drought_threshold)
        output[str(scale)] = {
            "scale_months": scale,
            "values": [{"date": d.strftime("%Y-%m-%d"), "spi": _clean_number(v)} for d, v in spi.items()],
            "drought_events": events,
        }
    return output


def _drought_events(spi: pd.Series, threshold: float) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    active: list[tuple[pd.Timestamp, float]] = []
    for date, value in spi.items():
        if pd.notna(value) and value <= threshold:
            active.append((date, float(value)))
        elif active:
            events.append(_event(active))
            active = []
    if active:
        events.append(_event(active))
    return events


def _event(points: list[tuple[pd.Timestamp, float]]) -> dict[str, Any]:
    return {"start": points[0][0].strftime("%Y-%m-%d"), "end": points[-1][0].strftime("%Y-%m-%d"),
            "duration_months": len(points), "minimum_spi": min(v for _, v in points),
            "mean_spi": sum(v for _, v in points) / len(points)}


def _frequency(daily: pd.Series) -> dict[str, Any] | None:
    annual_maxima: dict[str, list[dict[str, Any]]] = {}
    for duration in (1, 2, 3, 5):
        rolling = daily.rolling(duration, min_periods=duration).sum()
        maxima = rolling.groupby(rolling.index.year).max().dropna()
        rows = []
        for year, value in maxima.items():
            rows.append({"year": int(year), "maximum_mm": float(value)})
        levels = []
        if len(maxima) >= 8 and maxima.std() > 0:
            try:
                loc, scale = stats.gumbel_r.fit(maxima.to_numpy(dtype=float))
                for return_period in (2, 5, 10, 25, 50, 100):
                    quantile = 1 - 1 / return_period
                    levels.append({"return_period_years": return_period,
                                   "rainfall_mm": _clean_number(stats.gumbel_r.ppf(quantile, loc=loc, scale=scale))})
            except (ValueError, FloatingPointError):
                levels = []
        annual_maxima[str(duration)] = {"duration_days": duration, "annual_maxima": rows,
                                        "gumbel_return_levels": levels,
                                        "fit_note": "Exploratory Gumbel fit; interpret with the supplied record length."}
    return annual_maxima


def analyze_series(series: RainfallSeries, options: AnalysisOptions) -> dict[str, Any]:
    annual, _ = _annual_series(series, options)
    years = annual.index.astype(int).tolist()
    annual_values = [_clean_number(v) for v in annual.to_numpy(dtype=float)]
    monthly = _aggregate_monthly(series)
    monthly_rows = [{"date": d.strftime("%Y-%m-%d"), "year": int(d.year), "month": int(d.month), "rainfall_mm": _clean_number(v)}
                    for d, v in monthly.items()]

    if not monthly.empty:
        month_groups = monthly.groupby(monthly.index.month)
        season_map = {1: "winter", 2: "winter", 3: "pre-monsoon", 4: "pre-monsoon", 5: "pre-monsoon",
                      6: "southwest-monsoon", 7: "southwest-monsoon", 8: "southwest-monsoon", 9: "southwest-monsoon",
                      10: "post-monsoon", 11: "post-monsoon", 12: "post-monsoon"}
        seasonal = monthly.groupby([monthly.index.year, monthly.index.month.map(season_map)]).sum(min_count=1)
        seasonal_rows = [{"year": int(year), "season": str(season), "rainfall_mm": _clean_number(value)}
                         for (year, season), value in seasonal.items()]
        month_trends = {}
        for month, vals in month_groups:
            month_trends[str(int(month))] = trend_summary(vals.index.year.tolist(), [_clean_number(v) for v in vals.to_numpy()])
    else:
        seasonal_rows = []
        month_trends = {}

    valid = series.values.dropna()
    descriptive = {
        "observations": int(valid.size), "total_mm": _clean_number(valid.sum()),
        "mean_mm_per_observation": _clean_number(valid.mean()), "median_mm_per_observation": _clean_number(valid.median()),
        "std_mm_per_observation": _clean_number(valid.std(ddof=1)), "minimum_mm": _clean_number(valid.min()),
        "maximum_mm": _clean_number(valid.max()),
        "rainy_observations": int((valid >= options.rainy_day_mm).sum()),
        "wet_observations": int((valid >= options.wet_day_mm).sum()),
    }
    spi = _spi(monthly, options) if not monthly.empty else {}
    extremes = _extremes_daily(series, options)
    frequency = _frequency(series.values.asfreq("D")) if series.frequency == "daily" else None
    return _clean_value({
        "name": series.name, "source_format": series.source_format, "frequency": series.frequency,
        "unit": "mm", "record_start": valid.index.min().strftime("%Y-%m-%d"),
        "record_end": valid.index.max().strftime("%Y-%m-%d"), "descriptive": descriptive,
        "annual": [{"year": year, "rainfall_mm": value} for year, value in zip(years, annual_values)],
        "trend": trend_summary(years, annual_values), "monthly": monthly_rows,
        "seasons_imd": seasonal_rows, "trend_by_calendar_month": month_trends,
        "extreme_indices_daily": extremes, "spi": spi,
        "frequency_analysis": frequency,
        "notes": ["Uploaded series are processed in memory and are not written to disk.",
                  "Rainfall is treated as millimetres after the requested unit conversion.",
                  "SPI uses month-of-year gamma fits over the selected base period; scales with fewer than 10 calibration values may be null.",
                  "Extreme indices and frequency analysis require daily observations."],
    })


def analyze_payload(series: list[RainfallSeries], options: AnalysisOptions) -> dict[str, Any]:
    return {"generated_at": datetime.now(timezone.utc).isoformat(),
            "options": options.model_dump(),
            "series": [analyze_series(item, options) for item in series]}
