from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import re

import numpy as np
import pandas as pd

from .models import RainfallSeriesInput


UNIT_FACTORS = {"mm": 1.0, "cm": 10.0, "in": 25.4, "0.1mm": 0.1}
MISSING_TEXT = {"", "na", "n/a", "nan", "null", "-999", "-9999", "-99.9", "-999.0"}
MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7,
    "july": 7, "aug": 8, "august": 8, "sep": 9, "sept": 9,
    "september": 9, "oct": 10, "october": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


@dataclass
class RainfallSeries:
    name: str
    values: pd.Series
    frequency: str
    source_format: str


def _key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower()).strip()


def _find_column(frame: pd.DataFrame, aliases: set[str]) -> str | None:
    for column in frame.columns:
        if _key(column) in aliases:
            return str(column)
    return None


def _numeric(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    lowered = text.str.lower()
    numbers = pd.to_numeric(text.where(~lowered.isin({"t", "tr", "trace"}), "0"), errors="coerce")
    numbers = numbers.mask(lowered.isin(MISSING_TEXT))
    return numbers.mask(numbers < 0)


def _dates(values: pd.Series, dayfirst: bool = False) -> pd.DatetimeIndex:
    text = values.astype(str).str.strip()
    numeric = pd.to_numeric(text, errors="coerce")
    result = pd.to_datetime(text, errors="coerce", dayfirst=dayfirst)
    excel = numeric.between(1, 100_000) & result.isna()
    if excel.any():
        result.loc[excel] = pd.to_datetime(numeric.loc[excel], unit="D", origin="1899-12-30")
    return pd.DatetimeIndex(result)


def _make_series(name: str, dates: pd.Series | pd.DatetimeIndex, values: pd.Series, unit: str,
                 frequency: str = "auto", source_format: str = "csv") -> RainfallSeriesInput:
    index = pd.DatetimeIndex(dates)
    data = _numeric(values).to_numpy(dtype=float) * UNIT_FACTORS[unit]
    frame = pd.DataFrame({"value": data}, index=index)
    frame = frame.loc[~frame.index.isna()].sort_index()
    frame = frame.groupby(level=0, sort=True)["value"].sum(min_count=1)
    if frame.notna().sum() == 0:
        raise ValueError(f"{name}: no valid rainfall values were found")
    dates_out = [stamp.strftime("%Y-%m-%d") for stamp in frame.index]
    detected = frequency if frequency != "auto" else _infer_frequency(frame.index, dates)
    values_out = [float(value) if pd.notna(value) else None for value in frame.tolist()]
    return RainfallSeriesInput(name=name, dates=dates_out, values=values_out, unit="mm", frequency=detected,
                               source_format=source_format)


def _infer_frequency(index: pd.DatetimeIndex, raw_dates: object = None) -> str:
    if len(index) < 2:
        return "annual"
    if isinstance(raw_dates, pd.Series):
        sample = raw_dates.astype(str).str.strip()
        if sample.str.fullmatch(r"\d{4}").all():
            return "annual"
        if sample.str.fullmatch(r"\d{4}[-/]\d{1,2}").all():
            return "monthly"
    days = np.diff(index.asi8) / 86_400_000_000_000
    median = float(np.median(days))
    if median <= 2.0:
        return "daily"
    if median <= 40.0:
        return "monthly"
    if median <= 400.0:
        return "annual"
    return "irregular"


def from_payload(item: RainfallSeriesInput) -> RainfallSeries:
    dates = _dates(pd.Series(item.dates))
    values = pd.to_numeric(pd.Series(item.values), errors="coerce").astype(float)
    values = values.mask(values < 0) * UNIT_FACTORS[item.unit]
    frame = pd.DataFrame({"value": values.to_numpy()}, index=dates)
    frame = frame.loc[~frame.index.isna()].sort_index()
    frame = frame.groupby(level=0, sort=True)["value"].sum(min_count=1)
    if frame.notna().sum() < 2:
        raise ValueError(f"{item.name}: at least two valid dated observations are required")
    freq = item.frequency if item.frequency != "auto" else _infer_frequency(frame.index, pd.Series(item.dates))
    return RainfallSeries(item.name, frame, freq, item.source_format)


def parse_rainfall_csv(content: bytes, filename: str, unit: str = "mm", dayfirst: bool = False) -> list[RainfallSeriesInput]:
    if unit not in UNIT_FACTORS:
        raise ValueError("unit must be mm, cm, in, or 0.1mm")
    try:
        frame = pd.read_csv(BytesIO(content), dtype=str, keep_default_na=False)
    except Exception as exc:
        raise ValueError(f"Could not read CSV: {exc}") from exc
    if frame.empty or len(frame.columns) < 2:
        raise ValueError("The CSV needs a header row and at least two columns")
    frame.columns = [str(col).strip() for col in frame.columns]
    cols = {str(c): _key(c) for c in frame.columns}
    year = _find_column(frame, {"year", "yyyy"})
    month = _find_column(frame, {"month", "mon"})
    day = _find_column(frame, {"day", "dd"})
    date_col = _find_column(frame, {"date", "datetime", "time", "timestamp"})
    station = _find_column(frame, {"station", "station name", "site", "location"})
    value_col = _find_column(frame, {"rainfall", "rain", "precipitation", "precip", "value", "prcp", "mm"})
    if value_col is None:
        value_col = next((str(col) for col in frame.columns
                          if any(token in _key(col).split() for token in ("rainfall", "precipitation", "precip", "prcp"))), None)
    result: list[RainfallSeriesInput] = []

    if year and month and day and value_col:
        dates = pd.to_datetime(dict(year=pd.to_numeric(frame[year], errors="coerce"),
                                    month=pd.to_numeric(frame[month], errors="coerce"),
                                    day=pd.to_numeric(frame[day], errors="coerce")), errors="coerce")
        result.append(_make_series(value_col, dates, frame[value_col], unit, "daily", "year-month-day-value"))
    elif date_col and value_col and station:
        dates = _dates(frame[date_col], dayfirst)
        for label, sub in frame.groupby(station, dropna=False, sort=False):
            result.append(_make_series(str(label), dates[sub.index], sub[value_col], unit, "auto", "station-date-value"))
    elif date_col:
        dates = _dates(frame[date_col], dayfirst)
        candidates = [c for c in frame.columns if c != date_col]
        for col in candidates:
            nums = _numeric(frame[col])
            if nums.notna().any():
                result.append(_make_series(col, dates, nums, unit, "auto", "date-wide"))
    elif year and month and value_col:
        numeric_month = pd.to_numeric(frame[month], errors="coerce")
        month_numbers = frame[month].astype(str).str.strip().str.lower().map(MONTHS)
        numeric_month = numeric_month.fillna(month_numbers)
        dates = pd.to_datetime(dict(year=pd.to_numeric(frame[year], errors="coerce"),
                                    month=numeric_month, day=1), errors="coerce")
        result.append(_make_series(value_col, dates, frame[value_col], unit, "monthly", "year-month-value"))
    elif year:
        year_values = pd.to_numeric(frame[year], errors="coerce")
        month_columns = [c for c in frame.columns if _key(c) in MONTHS]
        day_columns = [c for c in frame.columns if re.fullmatch(r"(?:0?[1-9]|[12]\d|3[01])", _key(c))]
        if month_columns:
            dates_all, values_all = [], []
            for col in month_columns:
                month_number = MONTHS[_key(col)]
                dates_all.extend(pd.to_datetime({"year": year_values, "month": month_number, "day": 1}, errors="coerce").tolist())
                values_all.extend(frame[col].tolist())
            result.append(_make_series("Rainfall", pd.DatetimeIndex(dates_all), pd.Series(values_all), unit,
                                       "monthly", "year-month-columns"))
        elif month and day_columns:
            month_values = pd.to_numeric(frame[month], errors="coerce")
            for col in day_columns:
                day_number = int(_key(col))
                dates = pd.to_datetime({"year": year_values, "month": month_values, "day": day_number}, errors="coerce")
                result.append(_make_series(f"Day {day_number}", dates, frame[col], unit, "daily", "year-month-day-columns"))
        elif value_col:
            dates = pd.to_datetime(year_values.astype("Int64").astype(str), format="%Y", errors="coerce")
            result.append(_make_series(value_col, dates, frame[value_col], unit, "annual", "year-value"))
        else:
            for col in frame.columns:
                if col == year or _key(col) in {"month", "day"}:
                    continue
                vals = _numeric(frame[col])
                if vals.notna().any():
                    dates = pd.to_datetime(year_values.astype("Int64").astype(str), format="%Y", errors="coerce")
                    result.append(_make_series(col, dates, vals, unit, "annual", "year-wide"))

    if not result:
        raise ValueError("CSV layout not recognized. Use Date + station columns, Station/Date/Rainfall, Year/Month/Day/Rainfall, Year + Jan-Dec, Year/Month + day columns, or Year/Rainfall.")
    if len(result) > 20:
        raise ValueError("CSV produced more than 20 series; upload a smaller subset")
    return result


def materialize(payload: list[RainfallSeriesInput]) -> list[RainfallSeries]:
    return [from_payload(item) for item in payload]
