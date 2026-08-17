# ELMTEX coverage audit — Phase 0 decision gate

> **TEMPLATE — NOT YET RUN.** This audit requires cloning the ELMTEX corpus from the
> Fraunhofer GitLab referenced in [arXiv:2502.05638](https://arxiv.org/abs/2502.05638)
> (Task 5 Step 1) and running `scripts/audit_elmtex.py` against it (Task 5 Step 3), both
> of which need network access and a licence check this machine does not have. No part
> of this document below is a real result — every bracketed field is a placeholder to be
> filled in when the audit is actually run. Do not treat any number in this file as data.

## Fields to fill in once the audit runs

- **Date run:** `[YYYY-MM-DD]`
- **Corpus location:** `[path the ELMTEX clone was extracted to, e.g. data/raw/elmtex]`
- **Licence check:** `[licence name/link + one line confirming it permits this use — record
  this BEFORE running the audit, per Task 5 Step 1; if the licence does not permit use,
  stop here and record why]`
- **Raw key counts:** `[paste the "Found N distinct annotation keys" block and the full
  per-key count list printed by scripts/audit_elmtex.py]`
- **Per-target mapping table:** `[paste the "--- Mapping to our profile ---" block: for
  each of conditions/medications/allergies/vitals/procedures, MAPS or NO MATCH plus the
  matched key names]`
- **Result:** `[N]/5` resource types covered.
- **Chosen branch:**
  - If `N >= 3`: **ELMTEX mapper** — build a mapper from ELMTEX categories to our five
    resource types for real-test-set labelling.
  - If `N < 3`: **MTSamples fallback** — ELMTEX's categories don't cover enough of our
    profile; use MTSamples (or hand-labelling) for the real test set instead.
  - `[state which branch applies once N is known]`
- **Labelling-hours estimate under the chosen branch:** `[hours, with the reasoning: e.g.
  target real-test-set size (spec's "50 -> 200" ramp) x estimated minutes per record for
  the chosen branch]`

## Why this is a template, not a result

Per the controlling ruling for this work: every execution step in the Task 5 brief
(fetching ELMTEX, running the audit) is blocked on hardware/network/licence access not
available in this environment. Writing plausible-looking numbers into this file without
having run the audit would be fabrication, not a decision record — so this file
deliberately contains no `N/5` result, no key counts, and no branch decision yet. Fill in
every `[...]` field above only from the actual printed output of
`python scripts/audit_elmtex.py data/raw/elmtex`, run after Task 5 Step 1's licence check
passes.
