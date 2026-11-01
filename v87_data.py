"""Raw daily targets and strictly historical seasonal lag features."""
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = ("total", "bachelor", "master", "college", "other_degree", "budget", "other_funding", "government")
CATEGORIES = ("weekday", "phase", "saturday_slot")


def campaign(year):
    return pd.date_range(f"{year}-06-20", f"{year}-08-31", name="date")


def load_daily(directory):
    directory = Path(directory)
    rows = pd.read_csv(directory / "registered.csv", usecols=["abitur_kod", "reg_ts", "zayav_type_kod", "zayav_type_name", "priem_doc_type_kod"])
    funding = pd.read_csv(directory / "included_ks.csv", usecols=["abitur_kod", "reg_ts", "zayav_type_kod", "ob_osnova_name"])
    keys = ["abitur_kod", "date", "zayav_type_kod"]
    for frame in (rows, funding):
        frame["date"] = pd.to_datetime(frame.reg_ts, errors="raise").dt.normalize()
        if frame[keys].isna().any().any():
            raise ValueError("Missing registration date or matching key")
    if rows[["zayav_type_name", "priem_doc_type_kod"]].isna().any().any():
        raise ValueError("Missing education level or submission channel")
    funding["budget"] = funding.ob_osnova_name.isin(["Бюджет", "Гослиния"])
    rows = rows.merge(funding.groupby(keys).budget.max(), on=keys, how="left", validate="many_to_one")
    rows["budget"] = rows.budget.eq(True).astype(float)
    rows["other_funding"] = 1.0 - rows.budget
    degree = np.select([rows.zayav_type_name.str.startswith(prefix) for prefix in ("Бакалавр", "Магистр", "СПО")], TARGETS[1:4], default="other_degree")
    for name in TARGETS[1:5]:
        rows[name] = (degree == name).astype(float)
    rows["government"] = rows.priem_doc_type_kod.eq(4).astype(float)
    rows["total"] = 1.0
    years = sorted(rows.date.dt.year.unique())
    if not years or years != list(range(years[0], years[-1] + 1)):
        raise ValueError("Source seasons must be contiguous")
    index = pd.DatetimeIndex(np.concatenate([campaign(year).values for year in years]), name="date")
    return rows.groupby("date")[list(TARGETS)].sum().reindex(index, fill_value=0)


def history_before(counts, year):
    history = counts.loc[counts.index.year < year, list(TARGETS)].sort_index().copy()
    years = sorted(history.index.year.unique())
    if len(years) < 2 or years != list(range(years[0], year)):
        raise ValueError("Need contiguous preceding seasons")
    expected = pd.DatetimeIndex(np.concatenate([campaign(y).values for y in years]), name="date")
    if not history.index.equals(expected) or not np.isfinite(history.to_numpy()).all() or (history < 0).any().any():
        raise ValueError("Invalid or incomplete daily counts")
    if not np.allclose(history.total, history[list(TARGETS[1:5])].sum(axis=1)) or not np.allclose(history.total, history.budget + history.other_funding):
        raise ValueError("Target partitions do not reconcile")
    if (history.government > history.total).any():
        raise ValueError("Government channel exceeds total count")
    return history


def past_deadlines(history):
    result = {}
    for year in history.index.year.unique():
        for degree, fraction in (("bachelor", .20), ("master", .25)):
            series = history.loc[history.index.year == year, degree]
            threshold = fraction * series.loc[f"{year}-07-01":f"{year}-07-25"].max()
            if threshold <= 0:
                continue
            for date, count in series.loc[f"{year}-07-20":f"{year}-08-20"].items():
                after = series.loc[date + pd.Timedelta(days=1):date + pd.Timedelta(days=3)]
                if count >= threshold and len(after) == 3 and (after < threshold).all():
                    result[int(year), degree] = date
                    break
    return result


def calendar_features(dates, deadlines=None):
    dates = pd.DatetimeIndex(dates)
    start = pd.to_datetime(dates.year.astype(str) + "-06-20")
    position = (dates - start).days
    frame = pd.DataFrame({"position": position, "day": dates.day, "month": dates.month,
                          "weekday": dates.dayofweek.astype(str), "first_day": (position == 0).astype(int)}, index=dates)
    frame["workdays_from_start"] = np.busday_count(start.values.astype("datetime64[D]"), dates.values.astype("datetime64[D]"), weekmask="1111110")
    for degree, month_day in (("bachelor", "07-25"), ("master", "07-29")):
        nominal = pd.to_datetime(dates.year.astype(str) + "-" + month_day)
        if degree == "master":
            nominal -= pd.to_timedelta(np.maximum(nominal.dayofweek - 4, 0), unit="D")
        else:
            nominal = nominal.where(dates.year != 2021, pd.Timestamp("2021-07-29"))
        ends = pd.DatetimeIndex([(deadlines or {}).get((int(year), degree), end) for year, end in zip(dates.year, nominal)])
        frame[f"distance_{degree}"] = (dates - ends).days
        frame[f"after_{degree}"] = ((dates - ends).days > 0).astype(int)
        frame[f"end_weekday_{degree}"] = ends.dayofweek
    frame["phase"] = np.select([position < 4, dates.month == 6, frame.distance_bachelor <= 0,
                                 frame.distance_master <= 0], ["start", "june", "bachelor", "master"], default="tail")
    frame["saturday_slot"] = np.select([dates.dayofweek != 5, position < 7, (dates.month == 8) & (dates.day > 21), dates.month == 8],
                                       ["none", "first", "late", "august"], default="regular")
    return frame


def lag_features(dates, history):
    """Every lookup for a date in year y is explicitly restricted to years < y."""
    dates = pd.DatetimeIndex(dates)
    blocks = []
    # The exceptional postponed 2020 campaign is not a normal-season analog.
    observed = sorted(set(map(int, history.index.year)) - {2020})
    columns = ("total", "bachelor", "master", "college", "other_degree", "budget", "other_funding", "government")
    for year in sorted(set(dates.year)):
        target_dates = dates[dates.year == year]
        block = pd.DataFrame(index=target_dates)
        earlier = [past for past in observed if past < year]
        for lag in (1, 2):
            values = np.full((len(target_dates), len(columns), 4), np.nan)
            if len(earlier) >= lag:
                previous = earlier[-lag]
                source = history.loc[history.index.year == previous, list(columns)]
                same_date = pd.DatetimeIndex([date.replace(year=previous) for date in target_dates])
                shift = (target_dates.dayofweek - same_date.dayofweek + 3) % 7 - 3
                same_weekday = same_date + pd.to_timedelta(shift, unit="D")
                values[:, :, 0] = source.reindex(same_date).to_numpy()
                values[:, :, 1] = source.reindex(same_weekday).to_numpy()
                values[:, :, 2] = source.rolling(7, center=True, min_periods=1).mean().reindex(same_date).to_numpy()
                values[:, :, 3] = source.mean().to_numpy()[None, :]
            for group, column in enumerate(columns):
                for feature, name in enumerate(("calendar", "weekday", "week_mean", "level")):
                    block[f"lag{lag}_{column}_{name}"] = np.log1p(values[:, group, feature])
        blocks.append(block)
    return pd.concat(blocks).reindex(dates)
