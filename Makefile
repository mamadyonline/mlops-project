unit_test:
	pytest tests/

quality_checks:
	isort .
	ruff format
	ruff check --fix
	ruff check airflow/dags/ --select AIR3 --preview

