"""Raw admissions counts and calendar coordinates, independent of model versions."""

from pathlib import Path

import numpy as np
import pandas as pd

DEGREES = ("bachelor", "master", "background")
SLOTS = ("none", "first", "regular", "august", "late")


def campaign(year):
    return pd.date_range(f"{year}-06-20", f"{year}-08-31", name="date")


def load_counts(path):
    rows = pd.read_csv(path, usecols=["reg_ts", "zayav_type_name"])
    rows["date"] = pd.to_datetime(rows.reg_ts, errors="raise").dt.normalize()
    if rows[["date", "zayav_type_name"]].isna().any().any():
        raise ValueError("Missing registration date or education level")
    rows["degree"] = np.select(
        [rows.zayav_type_name.str.startswith(p) for p in ("Бакалавр", "Магистр")],
        DEGREES[:2],
        default="background",
    )
    years = sorted(rows.date.dt.year.unique())
    if years != list(range(years[0], years[-1] + 1)):
        raise ValueError("Campaign years must be contiguous")
    dates = pd.DatetimeIndex(np.concatenate([campaign(y).values for y in years]), name="date")
    result = rows.groupby(["date", "degree"]).size().unstack(fill_value=0)
    return result.reindex(index=dates, columns=DEGREES, fill_value=0).astype(float)


def history_before(counts, year):
    history = counts.loc[counts.index.year < year, list(DEGREES)].sort_index().copy()
    years = sorted(history.index.year.unique())
    if len(years) < 2 or years != list(range(years[0], year)):
        raise ValueError("Need contiguous preceding campaigns")
    dates = pd.DatetimeIndex(np.concatenate([campaign(y).values for y in years]), name="date")
    if (
        not history.index.equals(dates)
        or not np.isfinite(history.to_numpy()).all()
        or (history < 0).any().any()
    ):
        raise ValueError("Invalid or incomplete daily counts")
    return history.loc[history.index.year != 2020]


def load_exams(path):
    result = pd.read_csv(path).set_index("year")
    if not result.index.is_unique:
        raise ValueError("Duplicate exam years")
    result = result[["russian", "math", "social", "english"]].copy()
    for subject in result:
        result[subject] = pd.to_datetime(result[subject], errors="raise")
        if result[subject].isna().any() or not np.array_equal(
            result[subject].dt.year, result.index
        ):
            raise ValueError("Invalid exam dates")
    return result


def saturday_slots(dates):
    dates = pd.DatetimeIndex(dates)
    position = (dates - pd.to_datetime(dates.year.astype(str) + "-06-20")).days
    return np.select(
        [
            dates.dayofweek != 5,
            position < 7,
            (dates.month == 8) & (dates.day > 21),
            dates.month == 8,
        ],
        ["none", "first", "late", "august"],
        default="regular",
    )


def nominal_deadline(year, degree):
    if degree == "bachelor":
        return pd.Timestamp(f"{year}-07-29" if year == 2021 else f"{year}-07-25")
    if degree == "master":
        date = pd.Timestamp(f"{year}-07-29")
        return date - pd.Timedelta(days=max(date.dayofweek - 4, 0))
    return pd.Timestamp(f"{year}-08-15")


def infer_deadlines(history):
    """Last substantial day followed by three low-count days, in past campaigns."""
    result = {}
    for year in sorted(set(history.index.year)):
        for degree, fraction in (("bachelor", 0.20), ("master", 0.25)):
            series = history.loc[str(year), degree]
            threshold = fraction * series.loc[f"{year}-07-01":f"{year}-07-25"].max()
            if threshold <= 0:
                continue
            for date, count in series.loc[f"{year}-07-20":f"{year}-08-20"].items():
                following = series.loc[date + pd.Timedelta(days=1) : date + pd.Timedelta(days=3)]
                if count >= threshold and len(following) == 3 and (following < threshold).all():
                    result[int(year), degree] = date
                    break
    return result


def features(dates, degree, deadlines=None, coordinates="event", exams=None):
    dates = pd.DatetimeIndex(dates)
    start = pd.to_datetime(dates.year.astype(str) + "-06-20")
    position = (dates - start).days.to_numpy()
    end = pd.DatetimeIndex(
        [
            (deadlines or {}).get((int(y), degree), nominal_deadline(int(y), degree))
            for y in dates.year
        ]
    )
    distance = (dates - end).days.to_numpy()
    work_distance = np.busday_count(
        end.values.astype("datetime64[D]"), dates.values.astype("datetime64[D]"), weekmask="1111100"
    )
    result = pd.DataFrame(
        {
            "weekday": dates.dayofweek.astype(str),
            "saturday_slot": saturday_slots(dates),
            "opening_ramp": np.minimum(position, 14),
            "first_day": (position == 0).astype(int),
            "distance": distance,
            "work_distance": work_distance,
            "after_deadline": (distance > 0).astype(int),
            "last_day": (distance == 0).astype(int),
            "last_workday": ((distance <= 0) & (work_distance == 0)).astype(int),
            "last_three_days": ((distance >= -3) & (distance <= 0)).astype(int),
            "last_saturday": ((dates.dayofweek == 5) & (distance >= -3) & (distance <= 0)).astype(
                int
            ),
            "deadline_weekday": end.dayofweek.astype(str),
        },
        index=dates,
    )
    if coordinates == "calendar" or degree == "background":
        result["position"] = position
        result["day"] = dates.day
        result["month"] = dates.month
        for other in ("bachelor", "master"):
            other_end = pd.DatetimeIndex(
                [
                    (deadlines or {}).get((int(y), other), nominal_deadline(int(y), other))
                    for y in dates.year
                ]
            )
            result[f"distance_{other}"] = (dates - other_end).days
    if exams is not None and degree == "bachelor":
        for subject in exams:
            exam_dates = pd.DatetimeIndex(exams.loc[dates.year, subject])
            result[f"since_exam_{subject}"] = np.clip((dates - exam_dates).days, -14, 35)
    return result


def categorical_columns():
    return {
        "weekday": [str(i) for i in range(7)],
        "saturday_slot": list(SLOTS),
        "deadline_weekday": [str(i) for i in range(7)],
    }
