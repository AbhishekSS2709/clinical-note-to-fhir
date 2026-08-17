## Consolidated BLOCKED-ON-HARDWARE checklist

Everything below still needs to run on the GPU+Java-equipped server, **in this order**.
Items from earlier tasks (3, 6, 10) are included because Task 11 Step 7 and the README's
results table both depend on their output — this is the single order to execute
end-to-end, not just this batch's four tasks.

1. **[Task 3 carryover]** Re-verify `synthea.py`'s `parse_bundle` against genuine Synthea
   output. Run `./run_synthea -p 3000 -s 42 --exporter.fhir.export true` (needs Java) and
   confirm the parser handles real bundle shapes, not just the hand-authored fixture in
   `tests/fixtures/bundle_sample.json`.
2. **[Task 6]** Run the note-generation smoke step (20 encounters) then the full
   `make data` (needs vLLM + GPU; generator is `Qwen/Qwen3-14B-AWQ` per
   `configs/data.yaml`). Produces `data/interim/pairs.jsonl` and, via Task 4/8's
   `make data` split step, `data/processed/{train,val,test_synthetic}.jsonl`.
3. **[Task 10]** Run the five baseline eval commands to populate `outputs/eval/*.json`
   and establish the 8B zero-shot number Task 11's fine-tune must beat:
   ```
   python -m fhir_extract.eval --system regex
   python -m fhir_extract.eval --system llm --model Qwen/Qwen3-8B --shots 0
   python -m fhir_extract.eval --system llm --model Qwen/Qwen3-8B --shots 5
   python -m fhir_extract.eval --system llm --model Qwen/Qwen3-8B --shots 5 --constrained
   python -m fhir_extract.eval --system llm --model Qwen/Qwen3-32B --shots 5
   ```
4. **[Task 11 Step 1]** `nvidia-smi --query-gpu=name,memory.total --format=csv`; adjust
   `configs/train_qlora_8b.yaml` if the card is not ~24GB.
5. **[Task 11 Step 4]** Decode one training batch's labels
   (`tokenizer.decode([t for t in labels if t != -100])`) and confirm only the JSON
   answer is visible, not the note — fix the loss-masking mechanism before spending GPU
   hours if the note appears.
6. **[Task 11 Step 5]** Smoke-train on a 100-row slice of `train.jsonl`, 1 epoch, via
   `make train`. Confirm it completes without OOM and a checkpoint appears under
   `outputs/adapters/qlora-8b/`.
7. **[Task 11 Step 6]** Kill the full run mid-way (Ctrl-C after a checkpoint is written),
   re-run `make train`, and confirm the log shows resume-from-checkpoint rather than a
   restart at step 0.
8. **[Task 11 Step 7, corrected per Ruling I]** Full fine-tune, then merge, then evaluate
   the merged model:
   ```
   make train
   make merge
   python -m fhir_extract.eval --system tuned --model outputs/merged/qlora-8b --constrained
   ```
   Confirm `micro_f1` exceeds the Task 10 8B zero-shot baseline (item 3 above) before
   proceeding to deployment.
9. **[Task 12 Step 5]** `make serve &` then:
   ```
   curl -s -X POST localhost:8000/extract -H 'Content-Type: application/json' \
     -d '{"note":"62yo M with HTN and T2DM. BP 148/92, HR 82. Continue metformin 500mg BID. NKDA."}' \
     | python -m json.tool
   ```
   Confirm `tuned` differs from `base` (identical output means the merge silently failed).
10. **[Task 12 Step 6]** Benchmark tok/s at batch 1, 8, 32; measure p50/p95 latency and
    peak VRAM. Write `outputs/serving_benchmark.json`. Convert to cost per 1M notes using
    a real GPU hourly rate, and record which rate was used.
11. **[Task 13 Step 3]** Open `http://localhost:8000/` in a browser, click Extract,
    confirm both panes populate and visibly differ.
12. **[Task 13 Step 4]** Deploy publicly (free-tier GPU host, or the 4B variant, or a
    recorded GIF walkthrough plus a static page serving cached example outputs if no host
    is available). Replace the README's `_pending deployment_` line with the real URL.
13. **[Task 13]** Regenerate the README results table strictly from `outputs/eval/*.json`
    — replace every `_pending_` cell; no number may be typed by hand.
14. **[Task 5 Step 1]** Clone ELMTEX from the Fraunhofer GitLab referenced in
    [arXiv:2502.05638](https://arxiv.org/abs/2502.05638) into `data/raw/elmtex/`. Confirm
    the licence permits this use before proceeding. Do not commit the corpus.
15. **[Task 5 Step 3]** Run `python scripts/audit_elmtex.py data/raw/elmtex` and capture
    its output (key counts, per-target mapping, `N/5` result).
16. **[Task 5 Step 4]** Fill in every bracketed placeholder in
    `docs/decisions/elmtex-audit.md` with the real output from step 15, state the chosen
    branch (ELMTEX mapper if N>=3, MTSamples fallback otherwise), and record a
    labelling-hours estimate under that branch. Remove the "TEMPLATE — NOT YET RUN" notice
    once filled in.
