"""Build the self-contained, Russian-language Excel-to-CSV V93 notebook."""

import ast
import json
from pathlib import Path
import textwrap

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(source):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(source).strip()))


def code(source):
    cells.append(nbf.v4.new_code_cell(textwrap.dedent(source).strip()))


def definition(filename, name):
    source = (ROOT / filename).read_text()
    tree = ast.parse(source)
    node = next(node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name)
    start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    return "\n".join(source.splitlines()[start - 1:node.end_lineno])


md("""
# V93. Прогноз ежедневных регистраций абитуриентов

**Excel → подготовка данных → отдельное обучение потоков → прогноз → CSV.**

Ноутбук воспроизводит V93, для которой пользователь сообщил **Public RMSLE 0,39621**
на общем прогнозе 2024 года. Обучаются три потока: бакалавриат и специалитет,
магистратура, СПО и остальные программы. Общий прогноз — их сумма.

**Как запустить:** поместите рядом с ноутбуком папку `abitur_stage1`, выберите
ядро Python и выполните **Restart Kernel and Run All Cells**. Прогнозы появятся
в `output/v93_notebook`. Все функции, параметры и календарь находятся в ячейках:
другие Python-файлы проекта и готовые прогнозы не требуются.

Считаются **строки регистраций за день**, а не уникальные люди или зачисленные.
Период — 20 июня–31 августа, 73 даты. Проверенный результат относится к общему
прогнозу 2024 года; отдельные потоки отдельно на лидерборде не оценивались.
""")

md("""
## 1. Библиотеки

Для воспроизводимости используются версии библиотек V93. Если они ещё не установлены,
раскомментируйте строку установки и выполните её до следующих ячеек.
""")
code("""
# %pip install numpy==2.2.6 pandas==2.3.3 catboost==1.2.10 scipy==1.14.1 scikit-learn==1.7.2 openpyxl==3.1.5 matplotlib==3.10.9
""")
code("""
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from catboost import CatBoostRegressor
from scipy.linalg import solve
from sklearn.preprocessing import SplineTransformer
from IPython.display import display, FileLink

get_ipython().run_line_magic("matplotlib", "inline")
pd.set_option("display.max_columns", 12)
pd.set_option("display.precision", 4)
""")

md("""
## 2. Настройки запуска

Изменяемые настройки собраны в одной ячейке. `EXPORT_DEGREE` выбирает основной CSV:

| Значение | Содержание |
|---|---|
| `total` | Общая сумма трёх потоков |
| `bachelor` | Бакалавриат и специалитет |
| `master` | Магистратура |
| `background` | СПО и остальные программы |
| `all` | Три потока и общая сумма в одной таблице |

При `SAVE_SEPARATE_FILES = True` дополнительно сохраняются отдельные CSV каждого
потока. Для повторения проверенного прогноза оставьте `TARGET_YEAR = 2024`.
""")
code("""
TARGET_YEAR = 2024
INPUT_DIR = Path("abitur_stage1")
OUTPUT_DIR = Path("output/v93_notebook")
EXPORT_DEGREE = "total"
SAVE_SEPARATE_FILES = True

REGISTERED_FILE = INPUT_DIR / "Зарегистрированы (2014-2023).xlsx"
DEGREES = ("bachelor", "master", "background")
DEGREE_NAMES = {
    "bachelor": "Бакалавриат и специалитет",
    "master": "Магистратура",
    "background": "СПО и остальные программы",
}
SLOTS = ("none", "first", "regular", "august", "late")

if EXPORT_DEGREE not in ("total", *DEGREES, "all"):
    raise ValueError("Проверьте EXPORT_DEGREE: total, bachelor, master, background или all")
if not REGISTERED_FILE.is_file():
    raise FileNotFoundError(f"Не найден Excel: {REGISTERED_FILE.resolve()}")
""")

md("""
## 3. Чтение исходного Excel

Для V93 нужна выгрузка **«Зарегистрированы (2014–2023)»**, лист `Sheet1`.
Используются дата регистрации `reg_ts` и уровень образования `zayav_type_name`.
Выгрузки «Включены в КС» и «Статусы заявлений» относятся к другим событиям
приёмной кампании и в расчёте V93 не участвуют.

Исходный файл читается напрямую. Промежуточный `registered.csv` не требуется.
""")
code("""
raw = pd.read_excel(
    REGISTERED_FILE,
    sheet_name="Sheet1",
    usecols=["reg_ts", "zayav_type_name"],
    engine="openpyxl",
)

if raw[["reg_ts", "zayav_type_name"]].isna().any().any():
    raise ValueError("В исходном Excel пропущены дата регистрации или уровень образования")

print(f"Прочитано регистраций: {len(raw):,}".replace(",", " "))
display(raw.head(5))
""")

md("""
## 4. Ежедневные количества по потокам

Уровень определяется по названию программы. Дата приводится к началу дня,
после чего строки считаются по парам «дата, поток». Для каждого года строится
полная сетка с 20 июня по 31 августа: день без регистраций получает ноль.
Строки не удаляются как дубликаты по абитуриенту: цель V93 — количество регистраций.
""")
code(definition("admissions_calendar.py", "campaign"))
code("""
registrations = raw.copy()
registrations["date"] = pd.to_datetime(registrations["reg_ts"], errors="raise").dt.normalize()
registrations["degree"] = np.select(
    [registrations["zayav_type_name"].str.startswith(prefix) for prefix in ("Бакалавр", "Магистр")],
    DEGREES[:2],
    default="background",
)

source_years = sorted(registrations["date"].dt.year.unique())
if source_years != list(range(source_years[0], source_years[-1] + 1)):
    raise ValueError("В исходных данных отсутствует целый год")

date_grid = pd.DatetimeIndex(
    np.concatenate([campaign(year).values for year in source_years]), name="date"
)
daily_counts = (
    registrations.groupby(["date", "degree"])
    .size()
    .unstack(fill_value=0)
    .reindex(index=date_grid, columns=DEGREES, fill_value=0)
    .astype(float)
)

annual_counts = daily_counts.groupby(daily_counts.index.year).sum()
annual_counts["total"] = annual_counts.sum(axis=1)
display(annual_counts.rename(columns=DEGREE_NAMES | {"total": "Всего регистраций"}))
""")

md("""
## 5. Календарь ЕГЭ

Расстояния до дат ЕГЭ используются в модели бакалавриата. Даты из исходной
V93 встроены ниже, поэтому дополнительный входной CSV не нужен. Это даты
экзаменов, а не итоги экзаменов и не количество поступающих.

Таблица поддерживает сезоны 2014–2024. Для другого прогнозного года её нужно
дополнить расписанием соответствующего года. Источник календаря —
[архив ЕГЭ Санкт-Петербурга](https://www.ege.spb.ru/result/index.php?mode=ege2024&wave=1).
""")
exam_csv = "\n".join(
    ",".join(line.split(",")[:5])
    for line in (ROOT / "external_data/ege/exam_calendar.csv").read_text().splitlines()
)
code(f'''
EGE_DATES = """{exam_csv}"""
exam_source = pd.read_csv(StringIO(EGE_DATES))
exam_calendar = exam_source.set_index("year")[["russian", "math", "social", "english"]].copy()
for subject in exam_calendar:
    exam_calendar[subject] = pd.to_datetime(exam_calendar[subject], errors="raise")

if TARGET_YEAR not in exam_calendar.index:
    raise ValueError(f"Добавьте в календарь ЕГЭ расписание на {{TARGET_YEAR}} год")
display(exam_calendar.loc[[TARGET_YEAR]].rename(columns={{
    "russian": "Русский язык", "math": "Математика",
    "social": "Обществознание", "english": "Английский язык",
}}))
''')

md("""
## 6. Параметры V93

Бакалавриат и магистратура используют отдельные CatBoost, остальные программы —
сплайновую регрессию. Числа перенесены из проверенной конфигурации без округления.
Внутри этого ноутбука повторный поиск параметров не выполняется.

| Настройка | Смысл |
|---|---|
| `half_life` | Через сколько лет вес сезона уменьшается вдвое |
| `history_window` | Число последних нормальных сезонов для обучения; 0 — вся история |
| `level_method` | Отдельный уровень сезона: `none`, `mean` или `trend` |
| `level_window` | Число предыдущих сезонов для оценки уровня |
| `coordinates` | Общий календарь или расстояния до собственного срока приёма |
| `align_history` | Определять границы прошлых кампаний по историческим регистрациям |
| `positive_only` | Обучать профиль только на исторических днях с регистрациями |
| `depth`, `iterations`, `learning_rate`, `l2_leaf_reg` | Параметры CatBoost |
| `knots`, `penalty` | Число узлов и регуляризация сплайнов |

Параметры другой семьи модели, например `knots` у CatBoost, не используются.
""")
code(definition("v93_model.py", "DegreeConfig"))
parameters = json.loads((ROOT / "output/v93/selected_config.json").read_text())["degrees"]
code("PARAMETERS = " + repr(parameters) + "\n\nconfigs = {degree: DegreeConfig(**values) for degree, values in PARAMETERS.items()}\n\ndisplay(pd.DataFrame(PARAMETERS).rename(columns=DEGREE_NAMES))")

md("""
## 7. Календарь приёмной кампании

Субботы разделяются на типы: первая суббота кампании, обычная июльская,
августовская и поздняя августовская. Их рабочий график оценивается по истории.

Ориентиры V93: 25 июля для бакалавриата, 29 июля для магистратуры с переносом
на предыдущую пятницу при выходном, 15 августа для остальных программ.
В историческом 2021 году для бакалавриата используется 29 июля.
""")
code(definition("admissions_calendar.py", "saturday_slots") + "\n\n" + definition("admissions_calendar.py", "nominal_deadline"))

md("""
## 8. Исторические сроки и обучающие сезоны

При выравнивании истории ищется последний значимый день, после которого
три дня подряд поток остаётся низким. Это помогает сопоставлять последние
дни приёма в разных годах. Будущие данные в такой поиск не попадают.

Для обучения берутся только годы **до прогнозного**. Аномальный сезон 2020
исключается. Если задано окно истории, каждый поток оставляет нужное ему
число последних нормальных сезонов.
""")
code(definition("admissions_calendar.py", "infer_deadlines"))
code(definition("admissions_calendar.py", "history_before"))

md("""
## 9. Признаки одного дня

Признаки описывают день недели, начало кампании, календарное и рабочее
расстояние до срока, последние дни приёма и субботу перед сроком.
Координаты `event` используют собственный срок потока. Координаты `calendar`
дополнительно включают число месяца и положение относительно других сроков.
Для бакалавриата добавляются расстояния до дат ЕГЭ, ограниченные диапазоном
от −14 до 35 дней, как в V93.
""")
code(definition("admissions_calendar.py", "categorical_columns") + "\n\n" + definition("admissions_calendar.py", "features"))
code("""
display(features(
    campaign(TARGET_YEAR)[:5], "bachelor",
    coordinates=configs["bachelor"].coordinates,
    exams=exam_calendar if configs["bachelor"].exams else None,
).T)
""")

md(r"""
## 10. Уровень регистраций будущего сезона

Модель работает с $z=\ln(1+n)$, где $n$ — количество регистраций за день.
Уровень сезона — среднее $z$ по будним дням. При `trend` по последним сезонам
оценивается линейный тренд с регуляризацией наклона. Из исторической цели
уровень вычитается, а к прогнозу профиля прибавляется прогноз будущего уровня.

Так отдельно моделируются типичная интенсивность потока и распределение
регистраций внутри сезона. Будущая годовая сумма не задаётся вручную.
""")
code(definition("v93_model.py", "future_level"))

md("""
## 11. Матрица признаков для модели

CatBoost получает таблицу с категориальными признаками. Для остальных программ
строятся кубические сплайны и взаимодействия со днями недели. Регуляризация
ограничивает величину коэффициентов. Порядок столбцов сохранён из V93.
""")
code("""
def make_design(frame, config, fitted_spline=None):
    if config.family == "catboost":
        return frame, None

    coordinates = ["opening_ramp", "distance"]
    if "position" in frame:
        coordinates.append("position")
    values = frame[coordinates].to_numpy()

    if fitted_spline is None:
        knots = [np.linspace(0, 14, config.knots), np.linspace(-60, 50, config.knots)]
        if "position" in frame:
            knots.append(np.linspace(0, 72, config.knots))
        fitted_spline = SplineTransformer(
            knots=np.array(knots).T, degree=3, include_bias=False, extrapolation="linear"
        ).fit(values)

    smooth = fitted_spline.transform(values)
    encoded = frame.copy()
    for column, categories in categorical_columns().items():
        encoded[column] = pd.Categorical(encoded[column], categories=categories)
    encoded = pd.get_dummies(encoded, columns=list(categorical_columns()))

    relative = smooth[:, config.knots + 1:2 * (config.knots + 1)]
    weekday = np.column_stack([frame.weekday.eq(str(i)).to_numpy(float) for i in range(1, 6)])
    interaction = (relative[:, :, None] * weekday[:, None, :]).reshape(len(frame), -1)
    matrix = np.column_stack([
        np.ones((len(frame), 1)), smooth, encoded.to_numpy(float) / 35, interaction
    ])
    return matrix, fitted_spline
""")

md("""
## 12. Обучение одного потока

Функция получает только его историческую целевую переменную. Веса сезонов
убывают экспоненциально. Частота нулевых суббот считается по общему
историческому потоку. Воскресенья исключаются из обучения и закрываются
при прогнозе. Для сплайнов СПО дополнительно выбираются положительные дни.

В результате сохраняются обученный регрессор или коэффициенты сплайнов,
уровень будущего сезона и оценки рабочего графика.
""")
code("""
def train_degree(counts, degree, year, config, exam_dates):
    history = history_before(counts, year)
    if config.history_window:
        preceding = sorted(set(history.index.year))[-config.history_window:]
        history = history.loc[history.index.year.isin(preceding)].copy()

    training_years = sorted(set(map(int, history.index.year)))
    deadlines = infer_deadlines(history) if config.align_history else {}
    exams = None
    if config.exams and degree == "bachelor":
        if (set(training_years) | {year}) - set(exam_dates.index):
            raise ValueError("В календаре ЕГЭ не хватает нужного года")
        exams = exam_dates.loc[exam_dates.index <= year].copy()

    weights = np.exp2(-(year - history.index.year.to_numpy()) / config.half_life)
    slots = saturday_slots(history.index)
    total = history.sum(axis=1)
    closure_rates = {
        slot: float(np.average(total[slots == slot].eq(0), weights=weights[slots == slot]))
        for slot in ("first", "late", "august", "regular")
    }

    target = np.log1p(history[degree])
    weekdays = history.index.dayofweek < 5
    levels = target[weekdays].groupby(history.index.year[weekdays]).mean()
    level = future_level(levels, year, config)
    if config.level_method != "none":
        target -= history.index.year.map(levels).to_numpy()

    use = history.index.dayofweek != 6
    if config.positive_only:
        use &= history[degree].to_numpy() > 0

    frame = features(history.index, degree, deadlines, config.coordinates, exams)
    design, spline = make_design(frame, config)
    fitted = {
        "degree": degree, "config": config, "training_years": training_years,
        "exams": exams, "level": level, "closure_rates": closure_rates,
        "spline": spline,
    }

    if config.family == "catboost":
        regressor = CatBoostRegressor(
            iterations=config.iterations, depth=config.depth,
            learning_rate=config.learning_rate, l2_leaf_reg=config.l2_leaf_reg,
            loss_function=config.loss, cat_features=list(categorical_columns()),
            one_hot_max_size=16, random_seed=93001 + DEGREES.index(degree),
            thread_count=1, allow_const_label=True, verbose=False, allow_writing_files=False,
        )
        regressor.fit(design.loc[use], target.loc[use], sample_weight=weights[use])
        fitted["regressor"] = regressor
    else:
        matrix, response, weight = design[use], target.to_numpy()[use], weights[use]
        gram = np.einsum("ni,n,nj->ij", matrix, weight, matrix)
        penalty = np.full(matrix.shape[1], config.penalty)
        penalty[0] = 1e-8
        gram += np.diag(penalty)
        rhs = np.einsum("ni,n,n->i", matrix, weight, response)
        fitted["coefficients"] = solve(gram, rhs, assume_a="pos")

    return fitted
""")

md("""
## 13. Обучение трёх отдельных моделей

В этой ячейке выполняется обучение. Таблица показывает, какие сезоны реально
использовал каждый поток и какой уровень был получен из истории.
""")
code("""
started = time.perf_counter()
models = {}
training_summary = []

for degree in DEGREES:
    models[degree] = train_degree(daily_counts, degree, TARGET_YEAR, configs[degree], exam_calendar)
    fitted = models[degree]
    training_summary.append({
        "Поток": DEGREE_NAMES[degree],
        "Модель": configs[degree].family,
        "Сезоны обучения": ", ".join(map(str, fitted["training_years"])),
        "Уровень log1p": fitted["level"],
    })

print(f"Обучение завершено за {time.perf_counter() - started:.2f} с")
display(pd.DataFrame(training_summary).set_index("Поток"))
""")

md("""
## 14. Прогноз одного потока

Для будущего сезона используются календарные ориентиры, а не наблюдения
будущего года. К прогнозу профиля добавляется уровень сезона. Обратное
преобразование `expm1` возвращает количество регистраций. После этого
применяется оценённый по истории график закрытых дней.
""")
code("""
def predict_degree(fitted, year):
    config = fitted["config"]
    degree = fitted["degree"]
    dates = campaign(year)
    frame = features(dates, degree, coordinates=config.coordinates, exams=fitted["exams"])
    design, _ = make_design(frame, config, fitted["spline"])

    if config.family == "catboost":
        profile = fitted["regressor"].predict(design)
    else:
        profile = np.einsum("ni,i->n", design, fitted["coefficients"])

    log_counts = profile + fitted["level"]
    prediction = np.expm1(np.maximum(0.0, log_counts))
    slots = saturday_slots(dates)
    closed = (dates.dayofweek == 6) | np.array([
        fitted["closure_rates"].get(slot, 0) > 0.5 for slot in slots
    ])
    prediction[closed] = 0.0

    if not np.isfinite(prediction).all():
        raise ValueError("Модель вернула некорректный прогноз")
    return pd.Series(prediction, index=dates, name=degree)
""")

md("""
## 15. Общий прогноз и итоги потоков

Общая сумма включает три потока. Поэтому бакалавриат и магистратура вдвоём
не равны общему прогнозу: есть также СПО и остальные программы. Дробные
значения сохраняются как математическое ожидание; округлять их до целых
для конкурсного CSV не нужно.
""")
code("""
forecast = pd.concat([predict_degree(models[degree], TARGET_YEAR) for degree in DEGREES], axis=1)
forecast["forecast"] = forecast.sum(axis=1)
forecast = forecast.reset_index()

display(forecast.head(10).rename(columns=DEGREE_NAMES | {"date": "Дата", "forecast": "Всего"}))
totals = forecast[list(DEGREES) + ["forecast"]].sum().rename(
    index=DEGREE_NAMES | {"forecast": "Общий прогноз"}
)
display(totals.to_frame("Регистраций за период"))
""")

md("""
## 16. График прогноза

Первый график показывает общую ежедневную динамику, второй — вклад каждого
потока. Провалы до нуля соответствуют закрытым дням календарного правила.
""")
code("""
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, layout="constrained")
axes[0].plot(forecast["date"], forecast["forecast"], color="#244b74", linewidth=2)
axes[0].set_title(f"Общий прогноз регистраций на {TARGET_YEAR} год", loc="left")
axes[1].stackplot(
    forecast["date"], *[forecast[degree] for degree in DEGREES],
    labels=[DEGREE_NAMES[degree] for degree in DEGREES],
    colors=["#3b82b5", "#65ad99", "#e5bd68"],
)
axes[1].set_title("Распределение по уровням образования", loc="left")
axes[1].legend(loc="upper right", fontsize=9)
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
for axis in axes:
    axis.set_ylabel("Регистраций в день")
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.2)
plt.show()
""")

md("""
## 17. Сохранение CSV

Основной файл определяется настройкой `EXPORT_DEGREE`. Общий прогноз и
отдельные потоки имеют формат `date,forecast`, как у V93. Файл `components`
содержит все потоки вместе. Сохраняется шесть знаков после запятой.

Выходная папка отдельная: существующий проверенный прогноз проекта
не перезаписывается.
""")
code("""
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
prefix = f"forecast_{TARGET_YEAR}_v93"

suffix = "" if EXPORT_DEGREE == "total" else f"_{'components' if EXPORT_DEGREE == 'all' else EXPORT_DEGREE}"
selected_path = OUTPUT_DIR / f"{prefix}{suffix}.csv"
selected_column = "forecast" if EXPORT_DEGREE == "total" else EXPORT_DEGREE
selected_table = (
    forecast if EXPORT_DEGREE == "all"
    else forecast[["date", selected_column]].rename(columns={selected_column: "forecast"})
)
selected_table.to_csv(selected_path, index=False, float_format="%.6f")

written_files = {selected_path}
if SAVE_SEPARATE_FILES:
    for degree in DEGREES:
        path = OUTPUT_DIR / f"{prefix}_{degree}.csv"
        forecast[["date", degree]].rename(columns={degree: "forecast"}).to_csv(
            path, index=False, float_format="%.6f"
        )
        written_files.add(path)
    components_path = OUTPUT_DIR / f"{prefix}_components.csv"
    forecast.to_csv(components_path, index=False, float_format="%.6f")
    written_files.add(components_path)

print(f"Основной CSV: {selected_path}")
for path in sorted(written_files):
    display(FileLink(str(path)))
""")

md("""
## Происхождение настроек

Это перенос фиксированной V93 в ноутбук, а не новый подбор. Гиперпараметры
V93 ранее выбирались по близости к прогнозу V84. Веса моделей в этом ноутбуке
обучаются на исторических регистрациях; файлы V84 и старые прогнозы не читаются.
Источник выбора параметров этим не меняется. Сообщённый пользователем
Public RMSLE 0,39621 относится к исходному общему CSV V93 за 2024 год.
""")

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10.6"},
}
# Keep cells readable without changing the calculations.
try:
    import black
    for cell in notebook.cells:
        if cell.cell_type == "code" and cell.source and not cell.source.startswith("# %pip"):
            cell.source = black.format_str(cell.source, mode=black.Mode(line_length=95)).rstrip()
except ImportError:
    pass

nbf.validate(notebook)
destination = ROOT / "V93_Excel_to_CSV.ipynb"
nbf.write(notebook, destination)
print(f"Created {destination.name}: {len(cells)} cells")
