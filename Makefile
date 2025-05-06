coverage-report:
	pytest
	python tools/parse_coverage.py > docs/COVERAGE_GAPS.md 