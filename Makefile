.PHONY: data checks notebook test lint

data:
	uv run python -m scripts.download

checks:
	uv run python -m scripts.prepare

notebook:
	cd notebooks && uv run jupyter nbconvert --execute --to notebook --inplace ab_test.ipynb

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .
