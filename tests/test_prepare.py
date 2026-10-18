from pathlib import Path

import pandas as pd

from scripts.prepare import build_checks

FIXTURES = Path(__file__).parent / "fixtures"


def test_checks_dedup_items(tmp_path):
    out = tmp_path / "checks.parquet"
    assert build_checks(FIXTURES / "purchases.csv", out) == 3
    checks = pd.read_parquet(out).set_index("transaction_id")
    # три позиции чека t1 — один чек на 300, а не 900
    assert checks.loc["t1", "purchase_sum"] == 300
    assert checks.loc["t1", "n_items"] == 3
    assert checks["purchase_sum"].sum() == 650
    assert list(checks.columns) == ["client_id", "ts", "purchase_sum", "store_id", "n_items"]
