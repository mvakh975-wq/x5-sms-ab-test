"""Статистика A/B-теста с бинарной метрикой: проверка сплита, эффект, мощность, CUPED, сегменты."""

import numpy as np
import pandas as pd
from scipy import optimize, stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import (
    confint_proportions_2indep,
    proportion_effectsize,
    proportions_ztest,
)


def srm_pvalue(n_t: int, n_c: int, expected_share: float = 0.5) -> float:
    """Sample Ratio Mismatch: p-value хи-квадрат теста против ожидаемой доли тестовой группы."""
    total = n_t + n_c
    expected = [total * expected_share, total * (1 - expected_share)]
    return float(stats.chisquare([n_t, n_c], f_exp=expected).pvalue)


def diff_proportions(x_t: int, n_t: int, x_c: int, n_c: int, alpha: float = 0.05) -> dict:
    """Разница долей тест − контроль: z-тест и интервал Вальда."""
    p_t, p_c = x_t / n_t, x_c / n_c
    _, p_value = proportions_ztest([x_t, x_c], [n_t, n_c])
    low, high = confint_proportions_2indep(x_t, n_t, x_c, n_c, method="wald", alpha=alpha)
    return {
        "p_t": p_t,
        "p_c": p_c,
        "diff": p_t - p_c,
        "lift": p_t / p_c - 1,
        "ci_low": float(low),
        "ci_high": float(high),
        "p_value": float(p_value),
    }


def mde_proportions(
    p0: float, n_t: int, n_c: int, alpha: float = 0.05, power: float = 0.8
) -> float:
    """Минимальный обнаружимый прирост доли (в долях, не в п.п.) при заданных размерах групп."""
    effect = NormalIndPower().solve_power(nobs1=n_c, alpha=alpha, power=power, ratio=n_t / n_c)
    # эффект Коэна h монотонно растёт с приростом — ищем прирост, дающий нужный h
    return float(
        optimize.brentq(lambda d: proportion_effectsize(p0 + d, p0) - effect, 1e-9, 1 - p0)
    )


def cuped(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """CUPED: y − θ·(x − mean(x)), θ = cov(y, x) / var(x). Среднее y не меняется."""
    y, x = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    theta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    return y - theta * (x - x.mean())


def bootstrap_diff_ci(
    y_t: np.ndarray, y_c: np.ndarray, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """Перцентильный бутстрап-интервал разницы средних."""
    rng = np.random.default_rng(seed)
    y_t, y_c = np.asarray(y_t), np.asarray(y_c)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = rng.choice(y_t, y_t.size).mean() - rng.choice(y_c, y_c.size).mean()
    low, high = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def segment_effects(df: pd.DataFrame, by: str, alpha: float = 0.05) -> pd.DataFrame:
    """Эффект в каждом сегменте с поправкой Холма на множественные сравнения."""
    rows = []
    for segment, g in df.groupby(by, observed=True):
        t, c = g[g["treatment_flg"] == 1]["target"], g[g["treatment_flg"] == 0]["target"]
        r = diff_proportions(t.sum(), t.size, c.sum(), c.size, alpha=alpha)
        rows.append({"segment": segment, "n": len(g), **r})
    out = pd.DataFrame(rows)
    out["p_holm"] = multipletests(out["p_value"], method="holm")[1]
    return out
