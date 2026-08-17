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

`configs/data.yaml` (`generation` block) takes:

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

`configs/serve.yaml` takes `backend`/`base_url`/`api_key`/`structured_output_mode`
too, but no single `model` key — it needs two model identifiers (tuned and
base), not one; see "Two models, one endpoint" below.

**The served model id has not been confirmed.** `configs/data.yaml` ships
with `model: Qwen/Qwen3-8B` as a placeholder — this is a guess, not a
verified value. `OpenAIBackend` raises immediately if `model` is missing
from config, naming the config key and telling you to run
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

## Two models, one endpoint

`configs/serve.yaml` is different: the demo compares a *tuned* and a *base*
model side by side, so it needs two distinct model identifiers, not one.
Over an OpenAI-compatible endpoint that means two **model ids** against one
shared server, not two engines:

| key | meaning |
|---|---|
| `base_url` | shared endpoint for both models (`backend: openai`) |
| `openai_tuned_model` | model id for the fine-tuned side |
| `openai_base_model` | model id for the untouched-base side, must match an id from `GET <base_url>/models` |
| `tuned_model` | local model path for the fine-tuned side (`backend: vllm`) |
| `base_model` | local model path for the base side (`backend: vllm`) |

vLLM's OpenAI server can expose a LoRA adapter as its own model id via
`--lora-modules`, which is how one server serves both identities:

```
vllm serve Qwen/Qwen3-8B --port 8003 \
    --lora-modules fhir-tuned=outputs/adapters/qlora-8b
```

That serves the base weights under their own id (`Qwen/Qwen3-8B`) and the
adapter under `fhir-tuned`, both reachable at the same `base_url`. Set
`openai_tuned_model: fhir-tuned` and `openai_base_model: Qwen/Qwen3-8B` to
match. `serve.py` builds one `OpenAIBackend` per resolved model id (cached),
both pointed at the same `base_url`.

`backend: vllm` (the default) uses `tuned_model`/`base_model` and still
runs two in-process engines on a GPU, unchanged from before.

**Guard against configuring both slots to the same model.** If the resolved
tuned and base identifiers are equal — e.g. `openai_tuned_model` and
`openai_base_model` both left at the same placeholder, or no LoRA adapter
has been registered yet — `serve.py` logs a prominent warning at startup
naming the two config keys involved, because the comparison is then
meaningless (both panes render identical output). It does not raise; this
is a legitimate way to smoke-test the HTTP plumbing before a fine-tune
exists. The resolved ids are also returned in the `/extract` response
(`models: {tuned, base}`) and shown in the UI under each pane's heading, so
what was actually compared is never silently hidden.

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
