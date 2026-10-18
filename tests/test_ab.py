import numpy as np
import pandas as pd
import pytest

from src.ab import (
    bootstrap_diff_ci,
    cuped,
    diff_proportions,
    mde_proportions,
    segment_effects,
    srm_pvalue,
)


def test_srm_pvalue():
    assert srm_pvalue(500, 500) == pytest.approx(1.0)
    # хи-квадрат = 2 · 100² / 5000 = 4 → p = 0,0455
    assert srm_pvalue(5100, 4900) == pytest.approx(0.0455, abs=1e-4)


def test_diff_proportions_matches_manual_calculation():
    r = diff_proportions(60, 100, 50, 100)
    assert r["diff"] == pytest.approx(0.10)
    assert r["lift"] == pytest.approx(0.20)
    # z-тест с объединённой долей: z = 0,1 / sqrt(0,55·0,45·0,02) = 1,4213
    assert r["p_value"] == pytest.approx(0.1552, abs=1e-4)
    # интервал Вальда: 0,1 ± 1,96·sqrt(0,6·0,4/100 + 0,5·0,5/100)
    assert r["ci_low"] == pytest.approx(-0.0372, abs=1e-4)
    assert r["ci_high"] == pytest.approx(0.2372, abs=1e-4)


def test_mde_proportions():
    # p0 = 0,5, по 1000 в группе, α = 0,05, мощность 0,8 → около 6,3 п.п.
    assert mde_proportions(0.5, 1000, 1000) == pytest.approx(0.0626, abs=5e-4)
    assert mde_proportions(0.5, 4000, 4000) < mde_proportions(0.5, 1000, 1000)


def test_cuped_keeps_mean_and_reduces_variance():
    rng = np.random.default_rng(0)
    x = rng.normal(10, 2, 5000)
    y = x + rng.normal(0, 1, 5000)
    adj = cuped(y, x)
    assert adj.mean() == pytest.approx(y.mean())
    assert adj.var() < 0.3 * y.var()


def test_bootstrap_ci_is_deterministic_and_covers_true_diff():
    rng = np.random.default_rng(1)
    t = rng.binomial(1, 0.6, 4000)
    c = rng.binomial(1, 0.5, 4000)
    low, high = bootstrap_diff_ci(t, c, n_boot=1000, seed=42)
    assert (low, high) == bootstrap_diff_ci(t, c, n_boot=1000, seed=42)
    assert low < 0.1 < high


def test_segment_effects_with_holm_correction():
    rows = []
    for seg, p_t, p_c in (("A", 0.70, 0.50), ("B", 0.50, 0.50)):
        for flg, p in ((1, p_t), (0, p_c)):
            n = 1000
            k = int(p * n)
            rows += [(seg, flg, 1)] * k + [(seg, flg, 0)] * (n - k)
    df = pd.DataFrame(rows, columns=["segment", "treatment_flg", "target"])
    out = segment_effects(df, by="segment").set_index("segment")
    assert out.loc["A", "diff"] == pytest.approx(0.20)
    assert out.loc["B", "diff"] == pytest.approx(0.0)
    assert out.loc["A", "p_holm"] < 0.001
    assert out.loc["B", "p_holm"] == pytest.approx(1.0)
    assert (out["p_holm"] >= out["p_value"]).all()
