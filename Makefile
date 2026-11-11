.PHONY: setup test validate report clean

setup:            ## install dependencies
	pip install -r requirements.txt

test:             ## run the test suite (19 tests)
	pytest -q

validate:          ## synthetic check: does the pipeline recover a known injected effect?
	python -m order_effect_eval.cli validate --n-items 500

report:            ## render the real pilot report from data/pilot_results.csv (+ data/special_cases.csv)
	python -m order_effect_eval.cli report data/pilot_results.csv --special-cases data/special_cases.csv

clean:
	rm -rf reports/*.md __pycache__ order_effect_eval/__pycache__ tests/__pycache__ .pytest_cache
