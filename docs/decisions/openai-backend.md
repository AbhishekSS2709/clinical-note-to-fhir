# OpenAI-compatible HTTP backend

Adds a second way to run inference: an HTTP client against an
OpenAI-compatible endpoint (e.g. `vllm serve ... --port 8003`), alongside the
existing in-process `vllm.LLM(...)` path. This lets note generation, the
baselines, and the demo server run from a machine with no GPU and no vLLM
installed, by pointing at a model served elsewhere.

Both backends implement one interface (`LLMBackend.complete()`) in
`src/fhir_extract/llm_client.py`. `generate.py`, `baselines.py`, and
`serve.py` build a backend via `build_backend(cfg)` and never import `vllm`
or `openai` directly; both packages are lazy-imported inside the backend
constructors, so the module can be imported (and the test suite runs) with
neither installed.

## Config keys

`configs/data.yaml` (`generation` block) and `configs/serve.yaml` both take:

| key | meaning | default |
|---|---|---|
| `backend` | `openai` or `vllm` | `vllm` |
| `model` | model id, **must match** an id returned by `GET <base_url>/models` | none — required, no fallback |
| `base_url` | e.g. `http://localhost:8003/v1` | required for `openai` |
| `api_key` | sent as the bearer token; vLLM ignores it but the client needs a non-empty string | `EMPTY` |
| `structured_output_mode` | `guided_json` \| `json_schema` \| `none` | `guided_json` |
| `max_concurrency` | worker pool size for concurrent requests | `8` |
| `timeout` | per-request timeout (seconds) | `120` |
| `max_retries` | client-level retries before a request is given up on | `3` |

**The served model id has not been confirmed.** `configs/data.yaml` and
`configs/serve.yaml` ship with `model: Qwen/Qwen3-8B` as a placeholder — this
is a guess, not a verified value. `OpenAIBackend` raises immediately if
`model` is missing from config, naming the config key and telling you to run
`curl <base_url>/models`. Do the same before a real job:

```
python scripts/check_endpoint.py --base-url http://<host>:8003/v1
```

This prints every model id the endpoint actually serves, runs one trivial
completion, and runs one completion with the `ClinicalRecord` JSON schema to
confirm structured output round-trips (parses as JSON and validates against
the model). Pass `--model` / `--structured-output-mode` to override the
`configs/data.yaml` defaults for the check.

## Pointing the pipeline at an endpoint

Set in `configs/data.yaml`:

```yaml
generation:
  backend: openai
  model: <id from GET base_url/models>
  base_url: http://<host>:8003/v1
```

Then run `generate.py` / `eval.py` as before — no code changes needed.
`baselines.py`'s `LLMBaseline(model, shots, constrained, examples)` keeps
building an in-process vLLM backend when called with its original signature
(unchanged, for `eval.py` and existing callers); pass `backend=build_backend(cfg)`
explicitly to route a baseline through the HTTP endpoint instead.

`configs/serve.yaml` takes the same `backend`/`base_url`/`api_key`/`model`/
`structured_output_mode` keys. Caveat: the demo compares a *tuned* and a
*base* model side by side, which needs two distinct model paths. The
`openai` config block only names one `model`/`base_url`, so when
`backend: openai`, `serve.py` routes **both** the tuned and base slots
through that single endpoint/model — useful for proving the HTTP plumbing
and UI work without a GPU, but it will not show a real tuned-vs-base
comparison. Set `backend: vllm` (the default, using `tuned_model`/
`base_model`) to get the real two-model comparison; that still requires a
GPU.

## Structured output modes

Servers differ in how they accept a JSON schema for constrained decoding,
so it's configurable rather than hardcoded:

- **`guided_json`** (default) — sends `extra_body={"guided_json": schema}`.
  This is the long-standing vLLM OpenAI-server form; use it for a vLLM
  `--served-model-name` server without other constraints on the request body.
- **`json_schema`** — sends `response_format={"type": "json_schema",
  "json_schema": {"name": "clinical_record", "schema": schema}}`, the OpenAI
  Chat Completions structured-output form. Use it against a server that
  implements the OpenAI structured-output API rather than vLLM's
  `guided_json` extension.
- **`none`** — sends no structured-output hint at all. Use it to measure the
  true unconstrained baseline (see the `constrained` flag on `LLMBaseline`),
  or against a server that supports neither of the above.

In all cases the schema itself always comes from
`ClinicalRecord.model_json_schema()` — never a hand-written copy — so the
constrained-decoding contract can't drift from the profile in
`src/fhir_extract/profile.py`.

Concurrency: `OpenAIBackend.complete()` issues requests through a bounded
`ThreadPoolExecutor` (`max_concurrency` workers, default 8) so a 20k-prompt
generation batch doesn't run serially. Results are indexed back to their
input position, so they come back in prompt order regardless of completion
order. A single request that still fails after the client's retries yields
`""` for that prompt (with a logged warning) instead of aborting the batch.

## What remains GPU-only

- **Training** (`train.py`, QLoRA fine-tuning) — unaffected by this change,
  still requires a GPU via the `gpu` optional dependency group.
- **`scripts/merge_adapter.py`** (merging the LoRA adapter into base
  weights) — CPU-capable but part of the training/serving pipeline, not
  inference.
- The in-process `vllm` backend itself, obviously, still needs vLLM
  installed and local VRAM. It remains available (`backend: vllm`, the
  default) for anyone running on a GPU box.

Everything else — note generation, baselines (`eval.py`), and the demo
server — can now run entirely against an HTTP endpoint with no local GPU and
no `vllm` install.
