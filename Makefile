.PHONY: data checks notebook test lint

data:      ## скачать файлы X5 Retail Hero с Kaggle в data/raw/
	uv run python -m scripts.download

checks:    ## свернуть позиции чеков до уровня чека (data/interim/checks.parquet)
	uv run python -m scripts.prepare

notebook:  ## выполнить ноутбук, графики — в docs/
	cd notebooks && uv run jupyter nbconvert --execute --to notebook --inplace ab_test.ipynb

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .
