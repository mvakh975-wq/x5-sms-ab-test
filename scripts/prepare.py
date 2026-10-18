"""Сворачивает позиции чеков X5 (purchases.csv, ~4,5 ГБ) до уровня чека.

В исходном файле строка — позиция чека, а сумма чека purchase_sum повторяется в каждой позиции.
Pandas целиком такой файл не прочитает, поэтому свёртку делает DuckDB, дальше работа — в pandas.
"""

from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
CHECKS = ROOT / "data" / "interim" / "checks.parquet"


def build_checks(purchases_csv: Path, out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(
        f"""
        copy (
            select transaction_id,
                   any_value(client_id) as client_id,
                   any_value(transaction_datetime)::timestamp as ts,
                   any_value(purchase_sum)::double as purchase_sum,
                   any_value(store_id) as store_id,
                   count(*) as n_items
            from read_csv('{purchases_csv}', header = true, all_varchar = true)
            group by transaction_id
            order by ts
        ) to '{out}' (format parquet)
        """
    )
    n = con.execute(f"select count(*) from '{out}'").fetchone()[0]
    con.close()
    return n


def main() -> None:
    n = build_checks(RAW_DIR / "purchases.csv", CHECKS)
    print(f"Чеков: {n:,}".replace(",", " "))


if __name__ == "__main__":
    main()
