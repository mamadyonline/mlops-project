unit_test:
	pytest tests/

quality_checks:
	isort .
	ruff format
	ruff check --fix

