.PHONY: help install test lint data train eval serve

help:
	@echo "install  - install package + dev deps"
	@echo "test     - run pytest"
	@echo "data     - build the training corpus"
	@echo "train    - run QLoRA fine-tune"
	@echo "eval     - run eval harness"
	@echo "serve    - start FastAPI + vLLM"

install:
	pip install -e ".[dev]"

test:
	pytest -v

lint:
	ruff check src tests

data:
	python -m fhir_extract.dataset --config configs/data.yaml

train:
	python -m fhir_extract.train --config configs/train_qlora_8b.yaml

eval:
	python -m fhir_extract.eval --config configs/data.yaml

serve:
	uvicorn fhir_extract.serve:app --host 0.0.0.0 --port 8000
