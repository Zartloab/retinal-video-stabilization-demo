.PHONY: install lint test demo

install:
	pip install -r requirements.txt

lint:
	ruff check .
	black --check .

test:
	pytest -q

demo:
	bash scripts/run_demo.sh
