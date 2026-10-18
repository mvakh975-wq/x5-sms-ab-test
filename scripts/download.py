"""Скачивает клиентов, чеки и разметку эксперимента X5 Retail Hero с Kaggle в data/raw/.

Нужен аккаунт Kaggle: `uv run kaggle auth login` или токен в ~/.kaggle/.
"""

import subprocess
import zipfile
from pathlib import Path

DATASET = "mvyurchenko/x5-retail-hero"
FILES = ["clients.csv", "purchases.csv", "uplift_train.csv"]
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        if (RAW_DIR / name).exists():
            continue
        subprocess.run(
            ["kaggle", "datasets", "download", DATASET, "-f", name, "-p", str(RAW_DIR), "-q"],
            check=True,
        )
        archive = RAW_DIR / f"{name}.zip"
        if archive.exists():
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(RAW_DIR)
            archive.unlink()
    print("Данные в data/raw/")


if __name__ == "__main__":
    main()
