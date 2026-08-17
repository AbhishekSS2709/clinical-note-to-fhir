.PHONY: help install test lint data train merge eval serve

# Server run order: data -> train -> merge -> eval -> serve.
# `merge` must precede `eval`/`serve` for the tuned model -- vLLM cannot
# load a bare PEFT adapter directory as `model=`, only merged weights.
help:
	@echo "install  - install package + dev deps"
	@echo "test     - run pytest"
	@echo "data     - build the training corpus"
	@echo "train    - run QLoRA fine-tune"
	@echo "merge    - merge the LoRA adapter into base weights"
	@echo "eval     - run eval harness"
	@echo "serve    - start FastAPI + vLLM"
	@echo ""
	@echo "order: data -> train -> merge -> eval -> serve"

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

merge:
	python scripts/merge_adapter.py Qwen/Qwen3-8B outputs/adapters/qlora-8b outputs/merged/qlora-8b

eval:
	python -m fhir_extract.eval --config configs/data.yaml

serve:
	uvicorn fhir_extract.serve:app --host 0.0.0.0 --port 8000
