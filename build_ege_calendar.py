"""Extract exam dates, never result-publication dates, from public EGE archives."""
import html
import re
from pathlib import Path

import pandas as pd


def main():
    root = Path(__file__).resolve().parent / "external_data/ege"
    subjects = {
        "Русский язык": "russian", "Математика профильная": "math",
        "Обществознание": "social", "Английский язык": "english",
    }
    # Verified in the primary-source archives. These older live endpoints now
    # return an error page, so their search-indexed archive text was read manually.
    historical = {
        2014: ("05-29", "06-05", "06-11", "06-02"),
        2015: ("05-28", "06-04", "06-08", "06-11"),
        2016: ("05-30", "06-06", "06-08", "06-14"),
        2017: ("06-09", "06-02", "06-05", "06-13"),
        2018: ("06-06", "06-01", "06-14", "06-18"),
    }
    rows = []
    for year in range(2014, 2025):
        url = f"https://www.ege.spb.ru/result/index.php?mode=ege{year}&wave=1"
        if year in historical:
            dates = {name: f"{year}-{value}" for name, value in zip(subjects.values(), historical[year])}
            method = "manually_verified_primary_archive"
            if year in (2015, 2016):
                url = f"https://www.ege.spb.ru/result/index.php?mode=ege{year}&wave="
            if year == 2017:
                url = "https://www.ege.spb.ru/result/index.php?mode=ege2017"
            if year == 2014:
                url = "https://www.ege.spb.ru/index.php?Itemid=728&id=45%3Aantonpotyavin&limitstart=170&option=com_k2&task=user&view=itemlist"
        else:
            source = (root / f"ege_{year}.html").read_bytes().decode("cp1251")
            blocks = re.split(r'<div[^>]*class="[^"]*exam-date[^\"]*"[^>]*>', source)[1:]
            dates = {}
            for block in blocks:
                date = html.unescape(re.sub("<[^>]+>", "", block.split("</div>", 1)[0])).strip()
                match = re.fullmatch(r"(\d+) (мая|июня|июля) (\d{4})", date)
                if not match:
                    continue
                day, month, calendar_year = match.groups()
                if int(calendar_year) != year:
                    continue
                iso = f"{year}-{dict(мая='05', июня='06', июля='07')[month]}-{int(day):02}"
                for subject in re.findall(r'<div[^>]*class="[^"]*text-right[^\"]*"[^>]*>(.*?)</div>', block, re.S):
                    subject = html.unescape(re.sub("<[^>]+>", "", subject)).strip()
                    if subject in subjects:
                        key = subjects[subject]
                        dates[key] = min(dates.get(key, iso), iso)
            method = "parsed_local_primary_archive"
        if set(dates) != set(subjects.values()):
            raise ValueError(f"Incomplete schedule: {year}: {dates}")
        rows.append({"year": year, **dates, "source_url": url, "extraction": method})
    pd.DataFrame(rows).to_csv(root / "exam_calendar.csv", index=False)


if __name__ == "__main__":
    main()
