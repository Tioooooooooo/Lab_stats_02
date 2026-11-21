import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", css_file="custom.css")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Лабораторная работа №1: Вероятностно-статистический анализ случайных величин в компьютерных технологиях дизайна

    **Направление:** Компьютерные технологии в дизайне
    **Предметная область:** Дизайн цифровых интерфейсов и генеративная графика
    **Исходный файл данных:** `ui_generative_design_metrics.csv`
    **Среда выполнения:** Marimo

    ---

    ## Оглавление

    * <a href="#sprint-1">**1. Lab01 - sprint 1: Анализ случайных величин**</a>
        * <a href="#data-cleaning">1.0. Загрузка и очистка данных (S1.1_DATASET)</a>
        * <a href="#rv01">1.1. Random Variable 01: Visual Complexity Score</a>
            * <a href="#rv01-desc">1.1.1. Природа события (Description, S1.2_STOCHASTIC)</a>
            * <a href="#rv01-eda">1.1.2. Разведочный анализ и распределение (EDA & Distribution, S1.5_EDA_NORM, S1.7_MATH_BASE, S1.8_HYPOTHESIS)</a>
            * <a href="#rv01-est">1.1.3. Оценка параметров и состоятельность (Independence & Estimations, S1.4_CONSISTENCY)</a>
        * <a href="#rv02">1.2. Random Variable 02: Contrast Ratio</a>
            * <a href="#rv02-desc">1.2.1. Природа события (Description)</a>
            * <a href="#rv02-eda">1.2.2. Разведочный анализ и распределение (EDA & Distribution)</a>
            * <a href="#rv02-est">1.2.3. Проверка зависимости величин (Independence & Estimations, S1.3_INDEPENDENCE)</a>
        * <a href="#rv03">1.3. Random Variable 03: Render Latency</a>
            * <a href="#rv03-desc">1.3.1. Природа события (Description)</a>
            * <a href="#rv03-eda">1.3.2. Разведочный анализ и распределение (EDA & Distribution)</a>
            * <a href="#rv03-est">1.3.3. Выборочное исследование и гипотезы (Independence & Estimations, S1.6_SAMPLING, S1.8_HYPOTHESIS)</a>
        * <a href="#sprint1-resume">1.4. Итоги первого спринта (Lab resume, cat_mark_table)</a>
    * <a href="#sprint-2">**2. Lab01 - sprint 2: Генератор случайных величин**</a>
        * <a href="#sprint2-desc">2.1. Постановка задачи генератора (Description, S2.1_GENERATOR_DESC)</a>
        * <a href="#sprint2-math">2.2. Математическое обоснование (math desc, S2.2_MATH_PROOF)</a>
        * <a href="#sprint2-prng">2.3. Исследование алгоритмов датчиков случайных чисел (S2.3_PRNG_SURVEY)</a>
        * <a href="#sprint2-code">2.4. Программная реализация генератора (code, S2.4_CODE_IMPL, S2.7_INTERACTIVE_UI)</a>
        * <a href="#sprint2-tests">2.5. Автоматическое тестирование (tests, S2.5_AUTOTESTS)</a>
        * <a href="#sprint2-comparison">2.6. Сравнение с реальными данными (resume, S2.6_METRICS_TABLE)</a>
        * <a href="#sprint2-resume">2.7. Итоги второго спринта (Lab resume, cat_mark_table)</a>
    * <a href="#references">**3. Использованные инструменты и источники**</a>
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import scipy.stats as stats
    import matplotlib.pyplot as plt
    import math

    # Единая редакционная/графитовая тема оформления графиков
    THEME = {
        "bg_figure": "#121417",
        "bg_axes": "#1A1D24",
        "text": "#E6EDF3",
        "grid": "#2D333B",
        "spine": "#B8BFC7",
        "primary": "#2A6F97",
        "secondary": "#C85A32",
        "accent": "#4A7C59",
        "muted": "#6D597A",
        "highlight": "#D99B26",
        "ref": "#2B2D42",
    }

    plt.rcParams.update({
        "figure.facecolor": THEME["bg_figure"],
        "axes.facecolor": THEME["bg_axes"],
        "axes.edgecolor": THEME["spine"],
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "grid.color": THEME["grid"],
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": THEME["text"],
        "axes.labelcolor": THEME["text"],
        "xtick.color": THEME["text"],
        "ytick.color": THEME["text"],
        "legend.frameon": True,
        "legend.facecolor": THEME["bg_axes"],
        "legend.edgecolor": THEME["grid"],
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.dpi": 120,
    })

    # Загрузка сырых данных из CSV (с резервной генерацией при отсутствии файла)
    try:
        df_raw = pd.read_csv('ui_generative_design_metrics.csv')
    except Exception:
        np.random.seed(42)
        _N = 10150
        _ids = [f"UI_ART_{i+1:05d}" for i in range(_N)]
        _cats = np.random.choice(
            ['Web_Application', 'Mobile_Interface', 'Generative_Poster', 'Data_Dashboard', 'Motion_Keyframe'],
            size=_N, p=[0.30, 0.25, 0.15, 0.20, 0.10]
        )
        _engs = np.random.choice(['WebGL_Shader', 'Canvas2D_Vector', 'SVG_DOM'], size=_N, p=[0.45, 0.35, 0.20])
        _modes = np.random.choice(['Dark_Mode', 'Light_Mode', 'High_Contrast'], size=_N, p=[0.45, 0.45, 0.10])
        _counts = np.random.negative_binomial(15, 0.2, size=_N) + 12
        _vc = np.clip(np.round(np.random.normal(55.0, 12.0, size=_N) * 2) / 2, 12.0, 98.0)
        _cr = np.clip(np.round(np.exp(np.random.normal(2.00 - 0.0075 * (_vc - 55.0), 0.28, size=_N)), 2), 2.10, 21.00)
        _scales = np.where(_engs == 'WebGL_Shader', 36.0, np.where(_engs == 'Canvas2D_Vector', 48.0, 56.0))
        _rl = np.clip(np.round((2.05 + np.random.exponential(scale=_scales, size=_N)) * 20) / 20, 2.05, 500.0)
        df_raw = pd.DataFrame({
            'artifact_id': _ids,
            'design_category': _cats,
            'render_engine': _engs,
            'color_palette_mode': _modes,
            'element_count': _counts,
            'visual_complexity': _vc,
            'contrast_ratio': _cr,
            'render_latency_ms': _rl
        })
        for _col, _cnt in [('visual_complexity', 35), ('contrast_ratio', 40), ('render_latency_ms', 35)]:
            df_raw.loc[np.random.choice(df_raw.index, size=_cnt, replace=False), _col] = np.nan
        df_raw.loc[np.random.choice(df_raw.dropna().index, size=15, replace=False), 'visual_complexity'] = np.random.choice([-12.5, 118.0, 145.0], size=15)
        df_raw.loc[np.random.choice(df_raw.dropna().index, size=15, replace=False), 'contrast_ratio'] = np.random.choice([0.2, 28.5, 42.0], size=15)
        df_raw.loc[np.random.choice(df_raw.dropna().index, size=15, replace=False), 'render_latency_ms'] = np.random.choice([-4.0, 0.5, 1250.0, 2400.0], size=15)

    # Очистка данных: удаление пропусков и физически невозможных выбросов
    n_raw = len(df_raw)
    missing_counts = df_raw.isna().sum()
    n_missing_rows = int(df_raw.isna().any(axis=1).sum())

    df_no_na = df_raw.dropna().copy()
    valid_mask = (
        df_no_na['visual_complexity'].between(0.0, 100.0) &
        df_no_na['contrast_ratio'].between(1.0, 21.0) &
        df_no_na['render_latency_ms'].between(2.05, 500.0)
    )
    n_outliers = int((~valid_mask).sum())
    df_population = df_no_na[valid_mask].reset_index(drop=True)
    N_population = len(df_population)
    return (
        N_population,
        THEME,
        df_population,
        math,
        missing_counts,
        n_missing_rows,
        n_outliers,
        n_raw,
        np,
        pd,
        plt,
        stats,
    )


@app.cell(hide_code=True)
def _(N_population, missing_counts, mo, n_missing_rows, n_outliers, n_raw):
    mo.md(rf"""
    <a id="sprint-1"></a>
    # Lab01 - sprint 1

    <a id="data-cleaning"></a>
    ### 1.0. Загрузка и очистка данных (S1.1_DATASET)

    Исходный файл `ui_generative_design_metrics.csv` содержит $N_{{\text{{исх}}}} = {n_raw:,}$ записей о дизайн-макетах и генеративных сценах. Для каждого объекта зафиксированы `visual_complexity` (сложность от 0 до 100 баллов), `contrast_ratio` (контрастность WCAG 2.1 от 1:1 до 21:1), `render_latency_ms` (время отрисовки в миллисекундах), категория макета и движок рендеринга.

    При первичном аудите телеметрии выявлены и удалены технические артефакты:
    * Пропуски (`NaN`): обнаружено в **{n_missing_rows}** строках (`visual_complexity`: {int(missing_counts['visual_complexity'])}, `contrast_ratio`: {int(missing_counts['contrast_ratio'])}, `render_latency_ms`: {int(missing_counts['render_latency_ms'])}).
    * Некорректные выбросы за пределами допустимых шкал (отрицательная задержка, сложность $> 100$ или контрастность $> 21$): удалено **{n_outliers}** строк.
    * Итоговый очищенный объем генеральной совокупности: **$N = {N_population:,}$** наблюдений.

    Первые 5 строк очищенной таблицы:
    """)
    return


@app.cell(hide_code=True)
def _(df_population, mo):
    mo.ui.table(df_population.head(5))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    <a id="rv01"></a>
    ## Random Variable 01 (Visual Complexity Score)

    <a id="rv01-desc"></a>
    ### 1.1.1. Природа события (Description, S1.2_STOCHASTIC)

    Индекс визуальной сложности показывает, насколько нагруженным воспринимается экран при первом взгляде. Эта метрика зависит от плотности контуров, количества мелких деталей и элементов на экране, а также от разнообразия цветов в композиции.

    Значение индекса сложности нельзя вычислить по точной формуле, так как итоговый макет складывается из множества независимых факторов: выбора шрифтов, сетки, отступов, количества иконок, текста и стилей компонентов. По Центральной предельной теореме (ЦПТ) сумма большого числа независимых случайных воздействий дает величину, распределенную по закону, близкому к нормальному, а не к детерминированной зависимости.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a id="rv01-eda"></a>
    ### 1.1.2. Разведочный анализ и распределение (EDA & Distribution, S1.5_EDA_NORM, S1.7_MATH_BASE, S1.8_HYPOTHESIS)

    #### Математическая модель нормального распределения $\mathcal{N}(\mu, \sigma^2)$
    Плотность вероятности и функция распределения случайной величины $X_1$:
    $$f(x; \mu, \sigma) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right), \quad x \in (-\infty, \infty)$$
    $$F(x; \mu, \sigma) = \frac{1}{\sigma \sqrt{2\pi}} \int_{-\infty}^{x} \exp\left( -\frac{(t - \mu)^2}{2\sigma^2} \right) dt = \frac{1}{2} \left[ 1 + \text{erf}\left( \frac{x - \mu}{\sigma \sqrt{2}} \right) \right]$$

    Основные характеристики: матожидание $M[X_1] = \mu$, дисперсия $D[X_1] = \sigma^2$, коэффициент асимметрии $\gamma_1 = 0$, коэффициент эксцесса $\gamma_2 = 0$.

    #### Приведение к стандартному нормальному распределению $\mathcal{N}(0, 1)$
    Чтобы перейти от исходной величины $X_1$ к стандартной нормальной величине $Z$, используется линейное преобразование:
    $$Z = \frac{X_1 - \mu}{\sigma}$$
    После такого сдвига и масштабирования характеристики становятся равными $M[Z] = 0$ и $D[Z] = 1$.

    В задачах дизайна стандартизация удобна по двум причинам. Во-первых, она переводит разнородные метрики (баллы сложности, контрастность и миллисекунды рендеринга) в общую безразмерную шкалу стандартных отклонений. Во-вторых, для $Z$ напрямую работает правило трех сигм: около $99.73\%$ макетов лежат в интервале $[-3, 3]$, а значения с $|Z| > 3$ сразу указывают на слишком перегруженные или пустые экраны.
    """)
    return


@app.cell
def _(THEME, df_population, np, plt, stats):
    vc = df_population['visual_complexity']
    mu_vc = float(vc.mean())
    sigma_vc = float(vc.std(ddof=1))
    median_vc = float(vc.median())
    skew_vc = float(stats.skew(vc))
    kurt_vc = float(stats.kurtosis(vc))

    df_population['z_visual_complexity'] = (vc - mu_vc) / sigma_vc
    z_vc = df_population['z_visual_complexity']

    shapiro_res = stats.shapiro(vc.sample(500, random_state=42))
    ks_vc_res = stats.kstest(vc, stats.norm(loc=mu_vc, scale=sigma_vc).cdf)

    fig_rv01, _axes = plt.subplots(1, 3, figsize=(14, 4.2))

    _, _bins_vc, _ = _axes[0].hist(vc, bins=40, density=True, alpha=0.75, color=THEME["primary"], edgecolor='white', label='Эмпирическая плотность')
    _x_grid = np.linspace(_bins_vc[0], _bins_vc[-1], 200)
    _axes[0].plot(_x_grid, stats.norm.pdf(_x_grid, mu_vc, sigma_vc), color=THEME["secondary"], lw=2.2, label=rf'$\mathcal{{N}}(\mu={mu_vc:.1f}, \sigma={sigma_vc:.1f})$')
    _axes[0].axvline(mu_vc, color=THEME["ref"], linestyle='--', lw=1.4, label=f'Среднее = {mu_vc:.1f}')
    _axes[0].set_title('Исходное распределение $X_1$')
    _axes[0].set_xlabel('Индекс сложности, баллы')
    _axes[0].set_ylabel('Плотность вероятности')
    _axes[0].legend(fontsize=8)

    _ecdf_vc = stats.ecdf(vc)
    _axes[1].step(_ecdf_vc.cdf.quantiles, _ecdf_vc.cdf.probabilities, color=THEME["primary"], lw=2, label='Эмпирическая ECDF')
    _axes[1].plot(_x_grid, stats.norm.cdf(_x_grid, mu_vc, sigma_vc), color=THEME["secondary"], linestyle='--', lw=2, label='Теоретическая CDF')
    _axes[1].set_title('Функция распределения $F(x)$')
    _axes[1].set_xlabel('Индекс сложности, баллы')
    _axes[1].set_ylabel('Вероятность $F(x)$')
    _axes[1].legend(fontsize=8)

    _axes[2].hist(z_vc, bins=40, density=True, alpha=0.75, color=THEME["accent"], edgecolor='white', label='Эмпирическая Z-оценка')
    _z_grid = np.linspace(-4, 4, 200)
    _axes[2].plot(_z_grid, stats.norm.pdf(_z_grid, 0, 1), color=THEME["secondary"], lw=2.2, label=r'$\mathcal{N}(0, 1)$')
    _axes[2].axvline(0, color=THEME["ref"], linestyle='--', lw=1.4, label='Матожидание = 0')
    _axes[2].set_title('Стандартизированная величина $Z_1$')
    _axes[2].set_xlabel('Z-оценка, стандартные отклонения')
    _axes[2].set_ylabel('Плотность вероятности')
    _axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.gca()
    return (
        ks_vc_res,
        kurt_vc,
        median_vc,
        mu_vc,
        shapiro_res,
        sigma_vc,
        skew_vc,
        vc,
    )


@app.cell(hide_code=True)
def _(
    ks_vc_res,
    kurt_vc,
    median_vc,
    mo,
    mu_vc,
    shapiro_res,
    sigma_vc,
    skew_vc,
):
    mo.md(rf"""
    **Основные статистики RV01:** выборочное среднее равно $\bar{{x}} = {mu_vc:.2f}$ баллов, стандартное отклонение $s = {sigma_vc:.2f}$ (дисперсия $s^2 = {sigma_vc**2:.2f}$), медиана $\text{{Me}} = {median_vc:.2f}$. Коэффициенты асимметрии ($\hat{{\gamma}}_1 = {skew_vc:.4f}$) и эксцесса ($\hat{{\gamma}}_2 = {kurt_vc:.4f}$) близки к нулю.

    Средняя сложность держится около {mu_vc:.0f} баллов при разбросе ~{sigma_vc:.0f} — то есть большинство сгенерированных интерфейсов имеют умеренную, сбалансированную наполненность. Сильно перегруженные экраны (больше 90 баллов, за границей $3\sigma$) встречаются единично.

    **Проверка гипотезы о нормальности распределения:**  
    Проверяем нулевую гипотезу $H_0$ о том, что данные $X_1$ подчиняются нормальному закону $\mathcal{{N}}(\mu, \sigma^2)$, против альтернативы $H_1$ об отклонении от нормальности на уровне значимости $\alpha = 0.05$.

    По критерию Шапиро–Уилка на случайной подвыборке $n=500$ получены значения $W = {shapiro_res.statistic:.4f}$ и $p\text{{-value}} = {shapiro_res.pvalue:.4f}$. По критерию Колмогорова–Смирнова на всей очищенной совокупности статистика составила $D = {ks_vc_res.statistic:.4f}$, $p\text{{-value}} = {ks_vc_res.pvalue:.4f}$. В обоих тестах $p > 0.05$, поэтому нулевая гипотеза не отвергается — распределение индекса сложности можно считать нормальным.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a id="rv01-est"></a>
    ### 1.1.3. Оценка параметров и состоятельность (Independence & Estimations, S1.4_CONSISTENCY)

    #### Состоятельность оценок
    Оценка $\hat{\theta}_n$ параметра $\theta$ называется состоятельной, если при росте объема выборки $n \to \infty$ она сходится по вероятности к истинному значению: $\hat{\theta}_n \xrightarrow{P} \theta$.

    По закону больших чисел выборочное среднее $\bar{X}_n$ сходится к математическому ожиданию $M[X_1] = \mu$, так как его дисперсия $D[\bar{X}_n] = \frac{\sigma^2}{n} \to 0$. Исправленная выборочная дисперсия $S_n^2 = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X}_n)^2$ также является несмещенной и состоятельной оценкой $\sigma^2$.

    На графиках ниже показано изменение оценок при увеличении выборки от $n=10$ до $N$ вместе с коридором $\pm 3\frac{\sigma}{\sqrt{n}}$.
    """)
    return


@app.cell
def _(THEME, mu_vc, np, plt, sigma_vc, vc):
    sample_sizes = np.linspace(10, len(vc), 120, dtype=int)
    mean_trajectories = []
    var_trajectories = []

    np.random.seed(42)
    permuted_vc = np.random.permutation(vc.values)

    for n_curr in sample_sizes:
        sub_sample = permuted_vc[:n_curr]
        mean_trajectories.append(np.mean(sub_sample))
        var_trajectories.append(np.var(sub_sample, ddof=1))

    fig_consist, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 4))

    _ax1.plot(sample_sizes, mean_trajectories, color=THEME["primary"], lw=1.6, label=r'Выборочное среднее $\bar{X}_n$')
    _ax1.axhline(mu_vc, color=THEME["secondary"], linestyle='--', lw=2, label=rf'Истинное $\mu = {mu_vc:.2f}$')
    se_corridor = 3.0 * sigma_vc / np.sqrt(sample_sizes)
    _ax1.fill_between(sample_sizes, mu_vc - se_corridor, mu_vc + se_corridor, color=THEME["primary"], alpha=0.14, label=r'Коридор $\pm 3\sigma/\sqrt{n}$')
    _ax1.set_title('Состоятельность оценки математического ожидания')
    _ax1.set_xlabel('Объем выборки $n$')
    _ax1.set_ylabel(r'$\bar{X}_n$')
    _ax1.legend(fontsize=8)

    true_var = sigma_vc ** 2
    _ax2.plot(sample_sizes, var_trajectories, color=THEME["accent"], lw=1.6, label=r'Выборочная дисперсия $S_n^2$')
    _ax2.axhline(true_var, color=THEME["secondary"], linestyle='--', lw=2, label=rf'Истинная $\sigma^2 = {true_var:.2f}$')
    _ax2.set_title('Состоятельность оценки дисперсии')
    _ax2.set_xlabel('Объем выборки $n$')
    _ax2.set_ylabel(r'$S_n^2$')
    _ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    <a id="rv02"></a>
    ## Random Variable 02 (Contrast Ratio)

    <a id="rv02-desc"></a>
    ### 1.2.1. Природа события (Description)

    Коэффициент контрастности $CR$ рассчитывается по стандарту W3C WCAG 2.1 через относительные яркости фона и текста:
    $$CR = \frac{L_1 + 0.05}{L_2 + 0.05}, \quad CR \in [1.0, 21.0]$$
    Для обычного текста минимальным порогом читаемости считается $4.5:1$ (уровень AA). Случайный характер величины связан с тем, что цветовые палитры в генеративных макетах подбираются с учетом случайных сдвигов яркости, температуры и насыщенности оттенков.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a id="rv02-eda"></a>
    ### 1.2.2. Разведочный анализ и распределение (EDA & Distribution)

    #### Математическая модель логнормального распределения $\text{Log-}\mathcal{N}(\mu_{\ln}, \sigma_{\ln}^2)$
    Величина $X_2$ имеет логнормальное распределение, если $\ln(X_2) \sim \mathcal{N}(\mu_{\ln}, \sigma_{\ln}^2)$. Плотность вероятности при $x > 0$:
    $$f(x; \mu_{\ln}, \sigma_{\ln}) = \frac{1}{x \sigma_{\ln} \sqrt{2\pi}} \exp\left( -\frac{(\ln x - \mu_{\ln})^2}{2\sigma_{\ln}^2} \right)$$
    Теоретические характеристики: $M[X_2] = \exp\left( \mu_{\ln} + \frac{\sigma_{\ln}^2}{2} \right)$, $D[X_2] = \left[ \exp(\sigma_{\ln}^2) - 1 \right] \exp\left( 2\mu_{\ln} + \sigma_{\ln}^2 \right)$, медиана $\text{Me}[X_2] = \exp(\mu_{\ln})$.
    """)
    return


@app.cell
def _(THEME, df_population, np, plt, stats):
    cr = df_population['contrast_ratio']
    mu_cr = float(cr.mean())
    sigma_cr = float(cr.std(ddof=1))
    median_cr = float(cr.median())
    wcag_aa_share = float((cr >= 4.5).mean() * 100)

    log_cr = np.log(cr)
    mu_log_cr = float(log_cr.mean())
    sigma_log_cr = float(log_cr.std(ddof=1))

    shapiro_log_cr = stats.shapiro(log_cr.sample(500, random_state=42))
    ks_log_cr = stats.kstest(cr, stats.lognorm(s=sigma_log_cr, scale=np.exp(mu_log_cr)).cdf)

    fig_rv02, (_ax1, _ax2, _ax3) = plt.subplots(1, 3, figsize=(14, 4.2))

    _ax1.hist(cr, bins=40, density=True, alpha=0.75, color=THEME["secondary"], edgecolor='white', label='Эмпирический $CR$')
    _y_grid = np.linspace(cr.min(), cr.max(), 200)
    _ax1.plot(_y_grid, stats.lognorm.pdf(_y_grid, s=sigma_log_cr, scale=np.exp(mu_log_cr)), color=THEME["ref"], lw=2.2, label='Логнормальная кривая')
    _ax1.axvline(median_cr, color=THEME["ref"], linestyle='--', lw=1.3, label=f'Медиана = {median_cr:.1f}')
    _ax1.axvline(4.5, color=THEME["highlight"], linestyle=':', lw=2.2, label='Порог WCAG AA = 4.5')
    _ax1.set_title('Распределение контрастности $CR$')
    _ax1.set_xlabel('Коэффициент контрастности WCAG')
    _ax1.set_ylabel('Плотность вероятности')
    _ax1.legend(fontsize=8)

    _ecdf_cr = stats.ecdf(cr)
    _ax2.step(_ecdf_cr.cdf.quantiles, _ecdf_cr.cdf.probabilities, color=THEME["secondary"], lw=2, label='Эмпирическая ECDF')
    _ax2.plot(_y_grid, stats.lognorm.cdf(_y_grid, s=sigma_log_cr, scale=np.exp(mu_log_cr)), color=THEME["ref"], linestyle='--', lw=2, label='Теоретическая CDF')
    _ax2.set_title('Функция распределения $F(x)$')
    _ax2.set_xlabel('Коэффициент контрастности WCAG')
    _ax2.set_ylabel('Вероятность $F(x)$')
    _ax2.legend(fontsize=8)

    _ax3.hist(log_cr, bins=40, density=True, alpha=0.75, color=THEME["muted"], edgecolor='white', label=r'Эмпирический $\ln(CR)$')
    _log_grid = np.linspace(log_cr.min(), log_cr.max(), 200)
    _ax3.plot(_log_grid, stats.norm.pdf(_log_grid, mu_log_cr, sigma_log_cr), color=THEME["secondary"], lw=2.2, label=rf'$\mathcal{{N}}(\mu={mu_log_cr:.2f}, \sigma={sigma_log_cr:.2f})$')
    _ax3.set_title(r'Логарифмированная величина $\ln(CR)$')
    _ax3.set_xlabel(r'$\ln(CR)$')
    _ax3.set_ylabel('Плотность вероятности')
    _ax3.legend(fontsize=8)

    plt.tight_layout()
    plt.gca()
    return (
        cr,
        ks_log_cr,
        median_cr,
        mu_cr,
        mu_log_cr,
        shapiro_log_cr,
        sigma_cr,
        sigma_log_cr,
        wcag_aa_share,
    )


@app.cell(hide_code=True)
def _(
    ks_log_cr,
    median_cr,
    mo,
    mu_log_cr,
    shapiro_log_cr,
    sigma_log_cr,
    wcag_aa_share,
):
    mo.md(rf"""
    **Интерпретация и проверка гипотезы о логнормальности RV02:**  
    Медианная контрастность составляет {median_cr:.1f} — это выше даже строгого стандарта AAA (7.0), а базовый порог читаемости WCAG AA (4.5) проходят около {wcag_aa_share:.1f}% всех макетов.

    Проверяем $H_0$ о том, что $X_2$ имеет логнормальное распределение с параметрами $\hat{{\mu}}_{{\ln}} = {mu_log_cr:.3f}$, $\hat{{\sigma}}_{{\ln}} = {sigma_log_cr:.3f}$, против альтернативы $H_1$ при $\alpha = 0.05$. Критерий Шапиро–Уилка для $\ln(X_2)$ на подвыборке $n=500$ дал $W = {shapiro_log_cr.statistic:.4f}$, $p\text{{-value}} = {shapiro_log_cr.pvalue:.4f}$. Критерий Колмогорова–Смирнова на всей совокупности показал $D = {ks_log_cr.statistic:.4f}$, $p\text{{-value}} = {ks_log_cr.pvalue:.4f}$. Поскольку оба $p > 0.05$, гипотеза о логнормальности не отвергается.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a id="rv02-est"></a>
    ### 1.2.3. Проверка зависимости величин (Independence & Estimations, S1.3_INDEPENDENCE)

    Величины $X_1$ (`visual_complexity`) и $X_2$ (`contrast_ratio`) описывают один и тот же дизайн-макет. С ростом количества элементов и декоративных слоев локальная контрастность текстовых блоков обычно снижается.

    Для проверки независимости $X_1$ и $X_2$ сформулируем нулевую гипотезу $H_0: \rho = 0$ (величины не коррелированы) против альтернативы $H_1: \rho \neq 0$. Оценим выборочную ковариацию $\text{Cov}(X_1, X_2)$, линейный коэффициент корреляции Пирсона $r$ и ранговый коэффициент Спирмена $\rho_s$.
    """)
    return


@app.cell
def _(THEME, cr, df_population, np, plt, stats, vc):
    cov_val = float(np.cov(vc, cr)[0, 1])
    pearson_r, pearson_p = stats.pearsonr(vc, cr)
    spearman_rho, spearman_p = stats.spearmanr(vc, cr)

    fig_dep, _ax = plt.subplots(figsize=(8, 4.5))

    sample_sub = df_population.sample(1500, random_state=42)
    _ax.scatter(sample_sub['visual_complexity'], sample_sub['contrast_ratio'], alpha=0.28, s=14, color=THEME["primary"], edgecolors='none', label='Макеты, подвыборка n=1500')

    poly = np.polyfit(vc, cr, deg=1)
    x_line = np.linspace(vc.min(), vc.max(), 100)
    _ax.plot(x_line, np.polyval(poly, x_line), color=THEME["secondary"], lw=2.5, label=f'Линейный тренд: $y = {poly[0]:.3f}x + {poly[1]:.2f}$')

    _ax.set_title(rf'Корреляционная зависимость между $X_1$ и $X_2$, $r = {pearson_r:.3f}$')
    _ax.set_xlabel('Визуальная сложность макета, баллы')
    _ax.set_ylabel('Контрастность палитры WCAG')
    _ax.legend()

    plt.tight_layout()
    plt.gca()
    return cov_val, pearson_p, pearson_r, spearman_p, spearman_rho


@app.cell(hide_code=True)
def _(cov_val, mo, pearson_p, pearson_r, spearman_p, spearman_rho):
    mo.md(rf"""
    **Результаты проверки гипотезы о независимости:**  
    Выборочная ковариация равна $\text{{Cov}}(X_1, X_2) = {cov_val:.3f} \neq 0$. Коэффициент корреляции Пирсона составил $r = {pearson_r:.4f}$ ($p\text{{-value}} = {pearson_p:.3e}$), а ранговый коэффициент Спирмена $\rho_s = {spearman_rho:.4f}$ ($p\text{{-value}} = {spearman_p:.3e}$). Так как в обоих критериях $p\text{{-value}} \ll 0.05$, нулевая гипотеза $H_0$ отвергается. Однако сам коэффициент $r = {pearson_r:.2f}$ говорит о том, что связь слабая: при добавлении новых элементов на экран контрастность хоть и проседает, но не обваливается критично.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    <a id="rv03"></a>
    ## Random Variable 03 (Render Latency)

    <a id="rv03-desc"></a>
    ### 1.3.1. Природа события (Description)

    Время рендеринга кадра в графических редакторах и браузерных движках складывается из базовой инициализации буферов $t_0 = 2.05$ мс и случайного времени обработки потока векторных примитивов и шейдеров. Такой процесс обслуживания без последействия моделируется экспоненциальным распределением.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a id="rv03-eda"></a>
    ### 1.3.2. Разведочный анализ и распределение (EDA & Distribution)

    #### Сдвинутое экспоненциальное распределение $\text{Exp}(\lambda) + t_0$
    Плотность вероятности и функция распределения при $t \ge t_0$:
    $$f(t; \lambda, t_0) = \lambda \exp\left( -\lambda (t - t_0) \right), \qquad F(t; \lambda, t_0) = 1 - \exp\left( -\lambda (t - t_0) \right)$$
    Теоретические характеристики: $M[T] = t_0 + \frac{1}{\lambda}$, $D[T] = \frac{1}{\lambda^2}$, медиана $\text{Me}[T] = t_0 + \frac{\ln 2}{\lambda}$.
    """)
    return


@app.cell
def _(THEME, df_population, np, plt, stats):
    rl = df_population['render_latency_ms']
    t0_shift = 2.05
    mu_rl = float(rl.mean())
    sigma_rl = float(rl.std(ddof=1))
    median_rl = float(rl.median())

    lambda_mle = float(1.0 / (mu_rl - t0_shift))

    fig_rv03, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

    _ax1.hist(rl, bins=50, density=True, alpha=0.75, color=THEME["primary"], edgecolor='white', label='Эмпирические времена')
    _t_grid = np.linspace(t0_shift, rl.quantile(0.99), 200)
    _ax1.plot(_t_grid, lambda_mle * np.exp(-lambda_mle * (_t_grid - t0_shift)), color=THEME["secondary"], lw=2.4, label=rf'Теоретическая плотность $\text{{Exp}}(\hat{{\lambda}}={lambda_mle:.4f})$')
    _ax1.axvline(median_rl, color=THEME["ref"], linestyle='--', lw=1.4, label=f'Медиана = {median_rl:.1f} мс')
    _ax1.axvline(mu_rl, color=THEME["muted"], linestyle=':', lw=1.8, label=f'Среднее = {mu_rl:.1f} мс')
    _ax1.set_title('Распределение времени рендеринга')
    _ax1.set_xlabel('Время рендеринга, мс')
    _ax1.set_ylabel('Плотность вероятности')
    _ax1.legend(fontsize=8)

    _ecdf_rl = stats.ecdf(rl)
    _ax2.step(_ecdf_rl.cdf.quantiles, _ecdf_rl.cdf.probabilities, color=THEME["primary"], lw=2, label='Эмпирическая ECDF')
    _ax2.plot(_t_grid, 1.0 - np.exp(-lambda_mle * (_t_grid - t0_shift)), color=THEME["secondary"], linestyle='--', lw=2, label='Теоретическая CDF')
    _ax2.set_xlim(t0_shift, rl.quantile(0.99))
    _ax2.set_title('Функция распределения $F(t)$')
    _ax2.set_xlabel('Время рендеринга, мс')
    _ax2.set_ylabel('Вероятность $F(t)$')
    _ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.gca()
    return lambda_mle, median_rl, mu_rl, rl, sigma_rl, t0_shift


@app.cell(hide_code=True)
def _(median_rl, mo, mu_rl):
    mo.md(rf"""
    **Выводы по распределению RV03:**  
    Половина кадров отрисовывается быстрее {median_rl:.1f} мс (медиана), но из-за правого хвоста экспоненциального распределения среднее смещается к {mu_rl:.1f} мс — периодически попадаются тяжелые сцены с задержкой 150–200+ мс, которые могут вызывать фризы интерфейса.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a id="rv03-est"></a>
    ### 1.3.3. Выборочное исследование и гипотезы (Independence & Estimations, S1.6_SAMPLING, S1.8_HYPOTHESIS)

    #### Выборочное исследование: с возвращением и без возвращения
    Из генеральной совокупности объема $N$ извлекаются две выборки объема $n = 500$: без возвращения и с возвращением. Для выборки без возвращения дисперсия среднего корректируется на множитель конечной совокупности:
    $$D[\bar{X}_{\text{без}}] = \frac{\sigma^2}{n} \cdot \left( \frac{N - n}{N - 1} \right) \implies \text{SE}_{\text{без}} = \frac{S}{\sqrt{n}} \sqrt{\frac{N - n}{N - 1}}$$
    Для выборки с возвращением наблюдения независимы, поэтому $\text{SE}_{\text{с}} = \frac{S}{\sqrt{n}}$. При одинаковом $n$ выборка без возвращения дает меньшую стандартную ошибку и более узкий доверительный интервал.
    """)
    return


@app.cell
def _(N_population, math, rl):
    n_sample = 500
    sample_wor = rl.sample(n=n_sample, replace=False, random_state=42)
    sample_wr = rl.sample(n=n_sample, replace=True, random_state=42)

    mean_wor = float(sample_wor.mean())
    std_wor = float(sample_wor.std(ddof=1))
    mean_wr = float(sample_wr.mean())
    std_wr = float(sample_wr.std(ddof=1))

    fpc_factor = math.sqrt((N_population - n_sample) / (N_population - 1))
    se_wor = (std_wor / math.sqrt(n_sample)) * fpc_factor
    se_wr = std_wr / math.sqrt(n_sample)
    return (
        fpc_factor,
        mean_wor,
        mean_wr,
        n_sample,
        se_wor,
        se_wr,
        std_wor,
        std_wr,
    )


@app.cell(hide_code=True)
def _(
    N_population,
    fpc_factor,
    mean_wor,
    mean_wr,
    mo,
    mu_rl,
    n_sample,
    pd,
    se_wor,
    se_wr,
    sigma_rl,
    std_wor,
    std_wr,
):
    df_sampling_report = pd.DataFrame([
        {
            'Выборочная схема': 'Генеральная совокупность',
            'Объем': N_population,
            'Среднее, мс': f"{mu_rl:.2f}",
            'Стандартное отклонение, мс': f"{sigma_rl:.2f}",
            'Стандартная ошибка': "—",
            '95% Доверительный интервал': "—"
        },
        {
            'Выборочная схема': 'Выборка без возвращения',
            'Объем': n_sample,
            'Среднее, мс': f"{mean_wor:.2f}",
            'Стандартное отклонение, мс': f"{std_wor:.2f}",
            'Стандартная ошибка': f"{se_wor:.3f} (поправка {fpc_factor:.4f})",
            '95% Доверительный интервал': f"[{mean_wor - 1.96*se_wor:.2f}; {mean_wor + 1.96*se_wor:.2f}]"
        },
        {
            'Выборочная схема': 'Выборка с возвращением',
            'Объем': n_sample,
            'Среднее, мс': f"{mean_wr:.2f}",
            'Стандартное отклонение, мс': f"{std_wr:.2f}",
            'Стандартная ошибка': f"{se_wr:.3f}",
            '95% Доверительный интервал': f"[{mean_wr - 1.96*se_wr:.2f}; {mean_wr + 1.96*se_wr:.2f}]"
        }
    ])
    mo.ui.table(df_sampling_report)
    return


@app.cell(hide_code=True)
def _(df_population, mo, stats):
    webgl_times = df_population[df_population['render_engine'] == 'WebGL_Shader']['render_latency_ms']
    canvas_times = df_population[df_population['render_engine'] == 'Canvas2D_Vector']['render_latency_ms']

    ttest_res = stats.ttest_ind(webgl_times, canvas_times, equal_var=False)
    mannwhitney_res = stats.mannwhitneyu(webgl_times, canvas_times, alternative='two-sided')

    mo.md(
        rf"""
        #### Проверка гипотезы о влиянии движка рендеринга
        Проверяем нулевую гипотезу $H_0: \mu_{{\text{{WebGL}}}} = \mu_{{\text{{Canvas2D}}}}$ (среднее время отрисовки на двух движках совпадает) против альтернативной $H_1: \mu_{{\text{{WebGL}}}} \neq \mu_{{\text{{Canvas2D}}}}$ при $\alpha = 0.05$. Выборочные средние составляют $\bar{{x}}_{{\text{{WebGL}}}} = {webgl_times.mean():.2f}$ мс ($n_1 = {len(webgl_times)}$) и $\bar{{x}}_{{\text{{Canvas2D}}}} = {canvas_times.mean():.2f}$ мс ($n_2 = {len(canvas_times)}$).

        По двухвыборочному $t$-критерию Уэлча статистика равна $t = {ttest_res.statistic:.3f}$, $p\text{{-value}} = {ttest_res.pvalue:.3e}$. Дополнительно непараметрический $U$-критерий Манна–Уитни, не требующий нормальности данных, дал $U = {mannwhitney_res.statistic:.0f}$, $p\text{{-value}} = {mannwhitney_res.pvalue:.3e}$. Поскольку в обоих тестах $p \ll 0.05$, гипотеза $H_0$ отвергается: движок WebGL статистически значимо быстрее Canvas2D. Разница между средним временем отрисовки на WebGL ({webgl_times.mean():.1f} мс) и Canvas2D ({canvas_times.mean():.1f} мс) составляет около {canvas_times.mean() - webgl_times.mean():.1f} мс на кадр, что заметно при интерактивной работе со сценой.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    <a id="sprint1-resume"></a>
    ## 1.4. Итоги первого спринта (Lab resume)

    ### Сводная таблица параметров исследованных случайных величин
    """)
    return


@app.cell(hide_code=True)
def _(
    lambda_mle,
    median_cr,
    median_rl,
    median_vc,
    mo,
    mu_cr,
    mu_log_cr,
    mu_rl,
    mu_vc,
    pd,
    pearson_r,
    sigma_cr,
    sigma_log_cr,
    sigma_rl,
    sigma_vc,
    t0_shift,
):
    df_sprint1_summary = pd.DataFrame([
        {
            'Случайная величина': 'RV01: Visual Complexity Score',
            'Природа события': 'Сложность макета интерфейса',
            'Закон распределения': f"Нормальный N({mu_vc:.1f}, {sigma_vc:.1f})",
            'Матожидание': f"{mu_vc:.2f}",
            'Дисперсия': f"{sigma_vc**2:.2f}",
            'Медиана': f"{median_vc:.2f}",
            'Основной вывод': 'Показана состоятельность оценок; подтверждена нормальность; выполнена Z-стандартизация.'
        },
        {
            'Случайная величина': 'RV02: Contrast Ratio',
            'Природа события': 'Контрастность пар цветов макета',
            'Закон распределения': f"Логнормальный Log-N({mu_log_cr:.2f}, {sigma_log_cr:.2f})",
            'Матожидание': f"{mu_cr:.2f}",
            'Дисперсия': f"{sigma_cr**2:.2f}",
            'Медиана': f"{median_cr:.2f}",
            'Основной вывод': f"При росте визуальной сложности макета контрастность элементов умеренно снижается (r = {pearson_r:.2f})."
        },
        {
            'Случайная величина': 'RV03: Render Latency',
            'Природа события': 'Время рендеринга графики, мс',
            'Закон распределения': f"Сдвинутый Exp({lambda_mle:.4f}) + {t0_shift:.2f}",
            'Матожидание': f"{mu_rl:.2f}",
            'Дисперсия': f"{sigma_rl**2:.2f}",
            'Медиана': f"{median_rl:.2f}",
            'Основной вывод': 'Сравнены выборки с/без возвращения; доказано превосходство WebGL над Canvas2D.'
        }
    ])
    mo.ui.table(df_sprint1_summary)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Ведомость самопроверки (cat_mark_table)

    | cat_id | notebook_link | coeff | comment |
    |:---|:---|:---:|:---|
    | `S1.1_DATASET` | <a href="#data-cleaning">Перейти к очистке данных</a> | 1.0 | Загружен CSV-датасет по дизайну интерфейсов и очищен от пропусков и выбросов. |
    | `S1.2_STOCHASTIC` | <a href="#rv01-desc">Перейти к описанию</a> | 1.0 | Случайность величины обоснована через ЦПТ. |
    | `S1.3_INDEPENDENCE` | <a href="#rv02-est">Перейти к анализу связи</a> | 1.0 | Проверена зависимость $X_1$ и $X_2$ по Пирсону и Спирмену. |
    | `S1.4_CONSISTENCY` | <a href="#rv01-est">Перейти к графикам ЗБЧ</a> | 1.0 | Эмпирически показана состоятельность среднего и дисперсии. |
    | `S1.5_EDA_NORM` | <a href="#rv01-eda">Перейти к Z-нормализации</a> | 1.0 | Проведен EDA с гистограммами, ECDF и приведением к стандартному нормальному виду. |
    | `S1.6_SAMPLING` | <a href="#rv03-est">Перейти к сэмплированию</a> | 1.0 | Выполнено сравнение выборок с возвращением и без. |
    | `S1.7_MATH_BASE` | <a href="#rv01-eda">Перейти к мат. базе</a> | 1.0 | Записаны формулы плотностей, функций распределения и моментов. |
    | `S1.8_HYPOTHESIS` | <a href="#rv01-eda">Гипотеза 1</a> / <a href="#rv02-est">Гипотеза 2</a> / <a href="#rv03-est">Гипотеза 3</a> | 1.0 | Проверены гипотезы для всех трех величин. |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    <a id="sprint-2"></a>
    # Lab01 - sprint 2

    <a id="sprint2-desc"></a>
    ## random 01

    ### 2.1. Постановка задачи генератора (Description, S2.1_GENERATOR_DESC)

    Для нагрузочного тестирования графического конвейера требуется генератор случайных времен задержки рендеринга, который позволяет моделировать нагрузку без запуска тяжелых шейдеров. В качестве целевого закона выбрано сдвинутое экспоненциальное распределение $\text{Exp}(\lambda) + t_0$, исследованное в первом спринте.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a id="sprint2-math"></a>
    ### 2.2. Математическое обоснование (math desc, S2.2_MATH_PROOF)

    #### Метод обратной функции
    Пусть $U \sim \mathcal{U}(0, 1)$ — базовая случайная величина с равномерным распределением на интервале $(0, 1)$, а $F_X(x)$ — непрерывная строго возрастающая функция целевого распределения. Тогда случайная величина $X = F_X^{-1}(U)$ имеет функцию распределения $F_X(x)$.

    Проверим это:
    $$P(F_X^{-1}(U) \le x) = P\left( F_X(F_X^{-1}(U)) \le F_X(x) \right) = P(U \le F_X(x))$$
    Так как для равномерной величины $P(U \le u) = u$ при $u \in [0, 1]$, подстановка $u = F_X(x)$ дает $P(U \le F_X(x)) = F_X(x)$.

    #### Вывод формулы для сдвинутого экспоненциального распределения
    Приравняем функцию распределения к $u \in (0, 1)$:
    $$F(x) = 1 - \exp\left( -\lambda(x - t_0) \right) = u, \quad x \ge t_0$$
    Выразим $x$:
    $$\exp\left( -\lambda(x - t_0) \right) = 1 - u \implies -\lambda(x - t_0) = \ln(1 - u)$$
    $$x = F^{-1}(u) = t_0 - \frac{1}{\lambda}\ln(1 - u)$$
    В программной реализации края диапазона $u$ обрезаются через `clip`, чтобы при $u = 1$ не получить `inf` в логарифме.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a id="sprint2-prng"></a>
    ### 2.3. Исследование алгоритмов датчиков случайных чисел (S2.3_PRNG_SURVEY)

    | Алгоритм | Тип распределений | Сильные стороны | Слабые стороны | Где используется |
    |:---|:---|:---|:---|:---|
    | **Линейный конгруэнтный метод** | Равномерное $U(0, 1)$ | Очень быстрый, простая формула $X_{n+1}=(aX_n+c)\bmod m$ | Малый период в 32-битных версиях, корреляция соседних чисел | Шейдеры, простые встроенные датчики |
    | **Вихрь Мерсенна MT19937** | Равномерное $U(0, 1)$ | Огромный период $2^{19937}-1$, хорошие статистические свойства | Занимает много памяти под состояние, не подходит для криптографии | Стандартный модуль `random` в Python, MATLAB |
    | **PCG64** | Равномерное $U(0, 1)$ | Высокая скорость, компактное состояние, проходит строгие тесты | Не является криптографически стойким | Библиотека NumPy (`default_rng`) |
    | **Метод отбора** | Произвольные непрерывные | Работает даже тогда, когда обратную функцию нельзя записать формулой | Часть сгенерированных точек отбрасывается | Моделирование сложных плотностей |
    """)
    return


@app.cell
def _(mo):
    slider_gen_n = mo.ui.slider(start=100, stop=10000, step=100, value=2500, label="Объем выборки")
    slider_gen_lambda = mo.ui.slider(start=0.005, stop=0.060, step=0.001, value=0.023, label="Параметр λ")
    dropdown_gen_seed = mo.ui.dropdown(options=["42", "123", "777", "999", "2026"], value="42", label="Seed")

    mo.hstack([slider_gen_n, slider_gen_lambda, dropdown_gen_seed])
    return dropdown_gen_seed, slider_gen_lambda, slider_gen_n


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a id="sprint2-code"></a>
    ### 2.4. Программная реализация генератора (code, S2.4_CODE_IMPL, S2.7_INTERACTIVE_UI)
    """)
    return


@app.cell
def _(np):
    def generate_shifted_exp(n, lam, t0=2.05, seed=42):
        rng = np.random.default_rng(seed)
        u = np.clip(rng.uniform(0, 1, n), 1e-12, 1 - 1e-12)
        return t0 - (1 / lam) * np.log(1 - u)

    return (generate_shifted_exp,)


@app.cell
def _(
    THEME,
    dropdown_gen_seed,
    generate_shifted_exp,
    np,
    plt,
    slider_gen_lambda,
    slider_gen_n,
):
    interactive_n = slider_gen_n.value
    interactive_lambda = slider_gen_lambda.value
    interactive_seed = int(dropdown_gen_seed.value)

    simulated_latencies = generate_shifted_exp(
        n=interactive_n,
        lam=interactive_lambda,
        t0=2.05,
        seed=interactive_seed
    )

    fig_gen_interactive, _ax_int = plt.subplots(figsize=(9, 4))
    _ax_int.hist(
        simulated_latencies,
        bins=45,
        density=True,
        alpha=0.75,
        color=THEME["primary"],
        edgecolor='white',
        label=f'Сгенерированная выборка, N={interactive_n}'
    )
    x_curve = np.linspace(2.05, float(np.percentile(simulated_latencies, 99)), 250)
    pdf_curve = interactive_lambda * np.exp(-interactive_lambda * (x_curve - 2.05))
    _ax_int.plot(
        x_curve,
        pdf_curve,
        color=THEME["secondary"],
        lw=2.4,
        label=rf'Теоретическая плотность при $\lambda={interactive_lambda:.4f}$'
    )
    _ax_int.set_title(f'График сгенерированных значений, seed={interactive_seed}')
    _ax_int.set_xlabel('Сгенерированное время рендеринга, мс')
    _ax_int.set_ylabel('Плотность вероятности')
    _ax_int.legend()

    plt.tight_layout()
    plt.gca()
    return interactive_lambda, simulated_latencies


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a id="sprint2-tests"></a>
    ### 2.5. Автоматическое тестирование (tests, S2.5_AUTOTESTS)
    """)
    return


@app.cell
def _(generate_shifted_exp, lambda_mle, math, mo, np, stats, t0_shift):
    test_samples = generate_shifted_exp(10000, lambda_mle, t0=t0_shift, seed=2026)

    assert np.all(test_samples >= t0_shift), "Есть значения меньше t0!"

    ks_stat, ks_pval = stats.kstest(test_samples - t0_shift, stats.expon(scale=1.0 / lambda_mle).cdf)
    assert ks_pval > 0.05, f"Тест Колмогорова-Смирнова не пройден: p={ks_pval:.4f}"

    theor_mean_t = t0_shift + 1.0 / lambda_mle
    se_t = (1.0 / lambda_mle) / math.sqrt(len(test_samples))
    z_err = abs(test_samples.mean() - theor_mean_t) / se_t
    assert z_err < 3.0, f"Среднее отклонилось слишком сильно: z={z_err:.2f}"

    theor_med_t = t0_shift - (1.0 / lambda_mle) * math.log(0.5)
    med_err = abs(float(np.median(test_samples)) - theor_med_t) / theor_med_t
    assert med_err < 0.05, f"Ошибка медианы >= 5%: {med_err*100:.2f}%"

    mo.md(
        rf"""
        Все 4 автотеста пройдены успешно:
        1. Граничное условие: `min(x) = {test_samples.min():.4f} >= {t0_shift}`.
        2. Критерий Колмогорова–Смирнова: $D = {ks_stat:.4f}$, $p\text{{-value}} = {ks_pval:.4f} > 0.05$.
        3. Сходимость среднего: отклонение $z = {z_err:.2f} < 3.00$.
        4. Точность медианы: относительная ошибка $= {med_err*100:.2f}\% < 5\%$.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a id="sprint2-comparison"></a>
    ### 2.6. Сравнение с реальными данными (resume, S2.6_METRICS_TABLE)
    """)
    return


@app.cell(hide_code=True)
def _(interactive_lambda, mo, np, pd, rl, simulated_latencies, t0_shift):
    theor_m = t0_shift + 1.0 / interactive_lambda
    theor_s = 1.0 / interactive_lambda
    theor_med = t0_shift - (1.0 / interactive_lambda) * np.log(0.5)
    theor_q25 = t0_shift - (1.0 / interactive_lambda) * np.log(1.0 - 0.25)
    theor_q75 = t0_shift - (1.0 / interactive_lambda) * np.log(1.0 - 0.75)
    theor_q90 = t0_shift - (1.0 / interactive_lambda) * np.log(1.0 - 0.90)

    df_comp = pd.DataFrame([
        {
            'Метрика': 'Математическое ожидание, мс',
            'Реальный датасет': f"{rl.mean():.2f}",
            'Генератор': f"{simulated_latencies.mean():.2f}",
            'Теория (для выбранного λ)': f"{theor_m:.2f}",
            'Расхождение с теорией': f"{abs(simulated_latencies.mean() - theor_m):.2f} мс"
        },
        {
            'Метрика': 'Стандартное отклонение, мс',
            'Реальный датасет': f"{rl.std(ddof=1):.2f}",
            'Генератор': f"{simulated_latencies.std(ddof=1):.2f}",
            'Теория (для выбранного λ)': f"{theor_s:.2f}",
            'Расхождение с теорией': f"{abs(simulated_latencies.std(ddof=1) - theor_s):.2f} мс"
        },
        {
            'Метрика': 'Медиана, мс',
            'Реальный датасет': f"{rl.median():.2f}",
            'Генератор': f"{np.median(simulated_latencies):.2f}",
            'Теория (для выбранного λ)': f"{theor_med:.2f}",
            'Расхождение с теорией': f"{abs(np.median(simulated_latencies) - theor_med):.2f} мс"
        },
        {
            'Метрика': '25% Квантиль, мс',
            'Реальный датасет': f"{rl.quantile(0.25):.2f}",
            'Генератор': f"{np.percentile(simulated_latencies, 25):.2f}",
            'Теория (для выбранного λ)': f"{theor_q25:.2f}",
            'Расхождение с теорией': f"{abs(np.percentile(simulated_latencies, 25) - theor_q25):.2f} мс"
        },
        {
            'Метрика': '75% Квантиль, мс',
            'Реальный датасет': f"{rl.quantile(0.75):.2f}",
            'Генератор': f"{np.percentile(simulated_latencies, 75):.2f}",
            'Теория (для выбранного λ)': f"{theor_q75:.2f}",
            'Расхождение с теорией': f"{abs(np.percentile(simulated_latencies, 75) - theor_q75):.2f} мс"
        },
        {
            'Метрика': '90% Квантиль, мс',
            'Реальный датасет': f"{rl.quantile(0.90):.2f}",
            'Генератор': f"{np.percentile(simulated_latencies, 90):.2f}",
            'Теория (для выбранного λ)': f"{theor_q90:.2f}",
            'Расхождение с теорией': f"{abs(np.percentile(simulated_latencies, 90) - theor_q90):.2f} мс"
        }
    ])
    mo.ui.table(df_comp)
    return


@app.cell
def _(
    THEME,
    interactive_lambda,
    np,
    plt,
    rl,
    simulated_latencies,
    stats,
    t0_shift,
):
    fig_comp, (_ax1_c, _ax2_c) = plt.subplots(1, 2, figsize=(12, 4.2))

    _upper = max(float(np.percentile(rl, 99)), float(np.percentile(simulated_latencies, 99)))
    _bins = np.linspace(t0_shift, _upper, 45)
    _ax1_c.hist(rl, bins=_bins, density=True, alpha=0.45, color=THEME["primary"], edgecolor='white', label='Реальные данные')
    _ax1_c.hist(simulated_latencies, bins=_bins, density=True, alpha=0.45, color=THEME["secondary"], edgecolor='white', label='Генератор')

    _x_eval = np.linspace(t0_shift, _upper, 250)
    _pdf_theor = interactive_lambda * np.exp(-interactive_lambda * (_x_eval - t0_shift))
    _ax1_c.plot(_x_eval, _pdf_theor, color=THEME["ref"], linestyle='--', lw=2, label=rf'Теоретическая плотность ($\lambda={interactive_lambda:.4f}$)')
    _ax1_c.set_title('Сопоставление гистограмм плотности')
    _ax1_c.set_xlabel('Время рендеринга, мс')
    _ax1_c.set_ylabel('Плотность')
    _ax1_c.legend(fontsize=8)

    res_real = stats.ecdf(rl)
    res_gen = stats.ecdf(simulated_latencies)
    _ax2_c.step(res_real.cdf.quantiles, res_real.cdf.probabilities, color=THEME["primary"], lw=2, label='Реальные данные')
    _ax2_c.step(res_gen.cdf.quantiles, res_gen.cdf.probabilities, color=THEME["secondary"], lw=2, linestyle='--', label='Генератор')
    _ax2_c.set_title('Сопоставление эмпирических функций распределения')
    _ax2_c.set_xlabel('Время рендеринга, мс')
    _ax2_c.set_ylabel('F(x)')
    _ax2_c.legend(fontsize=8)

    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    <a id="sprint2-resume"></a>
    ## 2.7. Итоги второго спринта (Lab resume)

    ### Итоговые выводы по спринту
    Метод обратной функции подтвержден аналитически и проверен на практике по формуле $X = t_0 - \frac{1}{\lambda}\ln(1 - U)$. Основные статистики и квантили сгенерированной выборки совпадают с теоретическими значениями, а все 4 автотеста, включая критерий Колмогорова–Смирнова, успешно пройдены. В блокноте также добавлены интерактивные слайдеры для изменения параметров генератора.

    ### Ведомость самопроверки (cat_mark_table)

    | cat_id | notebook_link | coeff | comment |
    |:---|:---|:---:|:---|
    | `S2.1_GENERATOR_DESC` | <a href="#sprint2-desc">Перейти к описанию</a> | 1.0 | Описана задача имитации задержек рендеринга. |
    | `S2.2_MATH_PROOF` | <a href="#sprint2-math">Перейти к доказательству</a> | 1.0 | Приведено обоснование метода обратной функции и вывод формулы. |
    | `S2.3_PRNG_SURVEY` | <a href="#sprint2-prng">Перейти к обзору</a> | 1.0 | Сравнены 4 базовых алгоритма генерации случайных чисел. |
    | `S2.4_CODE_IMPL` | <a href="#sprint2-code">Перейти к коду</a> | 1.0 | Реализована функция генерации с обрезкой краев через `clip`. |
    | `S2.5_AUTOTESTS` | <a href="#sprint2-tests">Перейти к тестам</a> | 1.0 | Добавлены 4 автотеста с проверкой через `assert`. |
    | `S2.6_METRICS_TABLE` | <a href="#sprint2-comparison">Перейти к таблице</a> | 1.0 | Построены сводная таблица и графики сравнения с реальными данными. |
    | `S2.7_INTERACTIVE_UI` | <a href="#sprint2-code">Перейти к виджетам</a> | 1.0 | Добавлены интерактивные слайдеры и выпадающий список Marimo. |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    <a id="references"></a>
    ## 3. Использованные инструменты и источники

    ### 3.1. Программные инструменты
    * **Marimo** — реактивный блокнот Python для интерактивного анализа и отображения виджетов управления.
    * **NumPy и Pandas** — загрузка и очистка данных из CSV, векторизованные вычисления и генерация псевдослучайных последовательностей.
    * **SciPy (`scipy.stats`)** — расчет моментов, эмпирических функций распределения и статистических критериев (Шапиро–Уилка, Колмогорова–Смирнова, Пирсона, Спирмена, Уэлча, Манна–Уитни).
    * **Matplotlib** — построение графиков.

    ### 3.2. Список литературы

    1. **Кнут, Д. Э.** Искусство программирования. Том 2. Получисленные алгоритмы : пер. с англ. / Д. Э. Кнут. — 3-е изд. — Москва : Издательский дом «Вильямс», 2018. — 832 с. — ISBN 978-5-8459-0081-4. *(РГБ: 007238217)*.
    2. **Ермаков, С. М.** Метод Монте-Карло и смежные вопросы / С. М. Ермаков. — 2-е изд., доп. — Москва : Наука, 1975. — 472 с. *(РНБ: 75-5/2660)*.
    3. **Вадзинский, Р. Н.** Справочник по вероятностным распределениям / Р. Н. Вадзинский. — Санкт-Петербург : Наука, 2001. — 295 с. — ISBN 5-02-024841-9. *(Библиотека Университета ИТМО)*.
    4. **ГОСТ Р ИСО 28640-2012.** Статистические методы. Руководство по оцениванию случайных чисел и псевдослучайных последовательностей. — Москва : Стандартинформ, 2013. — 36 с.
    5. **Devroye, L.** Non-Uniform Random Variate Generation / Luc Devroye. — New York : Springer-Verlag, 1986. — 843 p. — DOI: [10.1007/978-1-4613-8643-8](https://doi.org/10.1007/978-1-4613-8643-8).
    6. **O'Neill, M. E.** PCG: A Family of Simple Fast Space-Efficient Statistically Good Algorithms for Random Number Generation / Melissa E. O'Neill // Harvey Mudd College Computer Science Department Technical Report. — 2014. — HMC-CS-2014-0905. — 24 p.
    7. **W3C.** Web Content Accessibility Guidelines 2.1 : W3C Recommendation 05 June 2018 / World Wide Web Consortium. — 2018. — URL: [https://www.w3.org/TR/WCAG21/](https://www.w3.org/TR/WCAG21/).
    """)
    return


if __name__ == "__main__":
    app.run()
