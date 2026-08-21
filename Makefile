.PHONY: help install test lint synthea generate data train train-qlora train-elmtex \
        serve-ab eval elmtex benchmark merge serve

# Full order: synthea -> generate -> data -> train -> serve-ab -> eval
#
# `merge` is OPTIONAL. vLLM loads a PEFT adapter directly via --enable-lora, and
# serving base + adapter from one process is what makes the A/B comparison share
# a server, sampling settings and parser. Merge only for single-model deployment.

help:
	@echo "install      - install package + dev deps"
	@echo "test         - run pytest"
	@echo ""
	@echo "synthea      - generate synthetic FHIR bundles (needs java + synthea jar)"
	@echo "generate     - write clinical notes from bundle subsets (needs an LLM endpoint)"
	@echo "data         - filter for faithfulness + split by patient"
	@echo ""
	@echo "train        - bf16 LoRA fine-tune (primary)"
	@echo "train-qlora  - 4-bit QLoRA fine-tune (precision ablation)"
	@echo "train-elmtex - fine-tune on real ELMTEX reports (v2)"
	@echo ""
	@echo "serve-ab     - vLLM serving base + adapters on one endpoint"
	@echo "eval         - baselines + fine-tuned, synthetic splits"
	@echo "elmtex       - fetch + convert the real-report evaluation set"
	@echo "benchmark    - serving throughput/latency/cost"
	@echo ""
	@echo "order: synthea -> generate -> data -> train -> serve-ab -> eval"

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check src tests

# --- corpus ---------------------------------------------------------------

# Needs java and synthea-with-dependencies.jar (MITRE, Apache 2.0).
# Population size comes from configs/data.yaml.
synthea:
	java -jar synthea-with-dependencies.jar -p 3000 --exporter.fhir.export true

# Requires an OpenAI-compatible endpoint at generation.base_url in the config.
generate:
	python -m fhir_extract.generate --config configs/data.yaml

data:
	python -m fhir_extract.dataset --config configs/data.yaml

# --- training -------------------------------------------------------------

train:
	python -m fhir_extract.train --config configs/train_lora_bf16_8b.yaml

train-qlora:
	python -m fhir_extract.train --config configs/train_qlora_8b.yaml

train-elmtex:
	python -m fhir_extract.train --config configs/train_elmtex_8b.yaml

# --- evaluation -----------------------------------------------------------

# Serves the base model and both adapters from one process, so every row of
# the results table shares server and sampling settings.
serve-ab:
	vllm serve $(BASE) --served-model-name qwen3-8b-base \
	  --enable-lora --max-lora-rank 16 \
	  --lora-modules fhir-lora=outputs/adapters/lora-bf16-8b/checkpoint-553 \
	  --host 0.0.0.0 --port 8004 --max-model-len 16384 --gpu-memory-utilization 0.85

eval:
	python -m fhir_extract.eval --config configs/data.yaml --system regex
	python -m fhir_extract.eval --config configs/data.yaml --system llm --shots 0
	python -m fhir_extract.eval --config configs/data.yaml --system llm --shots 5
	python -m fhir_extract.eval --config configs/data.yaml --system llm --shots 0 \
	  --no-schema-hint --model fhir-lora

# Real clinical reports (CC-BY-4.0). Scored on conditions,procedures only --
# see docs/decisions/elmtex-evaluation.md.
elmtex:
	mkdir -p data/raw/elmtex
	curl -sL -o data/raw/elmtex/test_en.json \
	  "https://zenodo.org/records/14793810/files/test_en.json?download=1"
	python -m fhir_extract.elmtex data/raw/elmtex/test_en.json \
	  --out data/processed/elmtex_test.jsonl --limit 600
	python -m fhir_extract.eval --config configs/data.yaml --split elmtex_test \
	  --system llm --shots 0 --no-schema-hint --model fhir-lora \
	  --resources conditions,procedures

benchmark:
	python scripts/benchmark_serving.py --base-url http://127.0.0.1:8004/v1

# Optional: single-model deployment without --enable-lora.
merge:
	python scripts/merge_adapter.py Qwen/Qwen3-8B \
	  outputs/adapters/lora-bf16-8b/checkpoint-553 outputs/merged/lora-bf16-8b

serve:
	uvicorn fhir_extract.serve:app --host 0.0.0.0 --port 8000
