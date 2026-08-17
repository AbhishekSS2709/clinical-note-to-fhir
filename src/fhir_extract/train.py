"""QLoRA fine-tune. Resumable by design -- server access is intermittent."""
import inspect
import logging
import json
import math
import re
from pathlib import Path
import typer
import yaml

logger = logging.getLogger(__name__)

app = typer.Typer()

# Marks the start of an assistant turn in Qwen's ChatML-style rendering.
# Used only as the response_template for the oldest TRL completion-only-loss
# API (DataCollatorForCompletionOnlyLM), which matches on rendered text
# rather than on chat-template {% generation %} markers.
_ASSISTANT_RESPONSE_TEMPLATE = "<|im_start|>assistant\n"

_ASSISTANT_BRANCH_RE = re.compile(
    r"\{%-?\s*(?:if|elif)\s+message(?:\[['\"]role['\"]\]|\.role)"
    r"\s*==\s*['\"]assistant['\"]\s*-?%\}"
)
_JINJA_TAG_RE = re.compile(r"\{%-?\s*(\w+)")


def ensure_generation_markers(tokenizer) -> bool:
    """SUPERSEDED -- replace with the server-verified implementation.

    This version was written OFFLINE against a guessed ChatML-shaped
    template and is known to be WRONG for Qwen3: it wraps the assistant
    if/elif *branch body*, but in Qwen3's real template that body emits
    the ``<|im_start|>assistant
`` header itself (in three branches), so
    the header lands INSIDE the supervised span. The header is prompt, not
    completion, and must be hoisted out first.

    A corrected, byte-identical-verified implementation lives on the GPU
    server at::

        /mnt/e_disk/wcte/abhishek/qwen3_8b/handoff/ensure_generation_markers.py

    Swap that in before training. It also gates its early-exit on a regex
    rather than a literal ``"{% generation %}"`` substring -- necessary
    because the patched markers use Jinja whitespace control
    (``{%- generation -%}``), which the substring check below would miss,
    breaking idempotency.

    Ensure `tokenizer.chat_template` wraps the assistant turn in
    `{% generation %}` / `{% endgeneration %}` markers.

    TRL's completion-only / assistant-only loss masking locates the
    response span by looking for these markers when it renders the chat
    template with `return_assistant_tokens_mask=True`. Qwen3's stock
    template does not contain them, so masking silently does nothing (or
    errors) unless this is patched first.

    Returns True once the template is confirmed to contain the markers
    (whether they were already present or were just inserted). Never
    returns without patching -- if the assistant branch can't be
    confidently located, this raises instead of silently no-opping, since
    a silent no-op here would reproduce the exact bug it exists to close.
    """
    template = getattr(tokenizer, "chat_template", None)
    if template is None:
        raise ValueError(
            "tokenizer.chat_template is None -- cannot insert {% generation %} "
            "markers, so completion-only loss masking cannot be verified. "
            "Set an explicit chat template on the tokenizer before training."
        )

    if "{% generation %}" in template:
        return True

    match = _ASSISTANT_BRANCH_RE.search(template)
    if not match:
        raise ValueError(
            "Could not find an assistant-role branch (a "
            "`{% if/elif message['role'] == 'assistant' %}` tag) in "
            "tokenizer.chat_template, so {% generation %} markers cannot be "
            "inserted and completion-only loss masking cannot be verified. "
            "Patch the chat template manually before training."
        )

    # Walk forward from the opening tag, tracking if/for nesting depth, to
    # find where this assistant branch ends (its elif/else/endif at the
    # same depth -- not one belonging to a nested if/for inside it).
    depth = 0
    end_of_branch = None
    for tag in _JINJA_TAG_RE.finditer(template, match.end()):
        keyword = tag.group(1)
        if keyword in ("if", "for"):
            depth += 1
        elif keyword in ("endif", "endfor"):
            if depth == 0:
                end_of_branch = tag.start()
                break
            depth -= 1
        elif keyword in ("elif", "else") and depth == 0:
            end_of_branch = tag.start()
            break

    if end_of_branch is None:
        raise ValueError(
            "Found an assistant-role branch in tokenizer.chat_template but "
            "could not find its closing elif/else/endif tag, so "
            "{% generation %} markers cannot be safely inserted. Patch the "
            "chat template manually before training."
        )

    body = template[match.end():end_of_branch]
    if not body.strip():
        raise ValueError(
            "The assistant-role branch in tokenizer.chat_template appears "
            "empty, so {% generation %} markers cannot be inserted "
            "confidently. Patch the chat template manually before training."
        )

    patched = (
        template[:match.end()]
        + "{% generation %}" + body + "{% endgeneration %}"
        + template[end_of_branch:]
    )
    tokenizer.chat_template = patched
    return True


def _first_supported_kwarg(target, candidates: list[str]) -> str:
    """Return the first name in `candidates` accepted by `target`'s
    signature (a callable, or a class -- its __init__ is inspected).

    Used to pick the right keyword argument across library versions
    (e.g. `dtype` vs `torch_dtype`, or which completion-only-loss flag
    exists) without depending on version-string parsing, which is more
    fragile than checking what the installed code actually accepts.
    """
    try:
        params = inspect.signature(target).parameters
    except (TypeError, ValueError):
        params = {}

    for name in candidates:
        if name in params:
            return name

    has_var_kwargs = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    if has_var_kwargs:
        return candidates[0]

    raise ValueError(
        f"None of {candidates} are accepted by "
        f"{getattr(target, '__qualname__', target)!r}. The installed "
        "library version is not one this compatibility shim was written "
        "for -- inspect its signature and update the candidate list."
    )


def resolve_dtype_kwarg(from_pretrained) -> str:
    """`torch_dtype=` was renamed to `dtype=` in transformers v5's
    `from_pretrained`. Return whichever name the installed version's
    signature actually accepts, preferring the new name."""
    return _first_supported_kwarg(from_pretrained, ["dtype", "torch_dtype"])


def resolve_warmup_kwargs(training_cfg: dict, target=None,
                          total_steps: int | None = None) -> dict:
    """Return a copy of `training_cfg` with at most one of
    `warmup_ratio` / `warmup_steps` present -- passing both to
    TrainingArguments/SFTConfig errors on some transformers/trl versions.

    `warmup_ratio` (see configs/train_qlora_8b.yaml) is the source of
    truth; if both are somehow set, `warmup_steps` is dropped. If
    `target` (the TrainingArguments/SFTConfig class or its __init__) is
    given, also verify it actually accepts whichever key survives.
    """
    cfg = dict(training_cfg)
    if "warmup_ratio" in cfg and "warmup_steps" in cfg:
        del cfg["warmup_steps"]

    if target is None:
        return cfg

    def _accepts(name: str) -> bool:
        try:
            _first_supported_kwarg(target, [name])
            return True
        except Exception:
            return False

    # transformers 5.x REMOVED warmup_ratio (verified on 5.15.0). Convert to
    # warmup_steps rather than failing -- warmup is a schedule detail, not a
    # reason to abort a training run.
    if "warmup_ratio" in cfg and not _accepts("warmup_ratio"):
        ratio = cfg.pop("warmup_ratio")
        if _accepts("warmup_steps"):
            if total_steps:
                cfg["warmup_steps"] = max(1, math.ceil(ratio * total_steps))
            else:
                raise ValueError(
                    "target does not accept 'warmup_ratio' (transformers 5.x "
                    "removed it) and total_steps was not supplied, so it cannot "
                    "be converted to 'warmup_steps'. Pass total_steps=... to "
                    "resolve_warmup_kwargs()."
                )
        else:
            logger.warning(
                "target accepts neither 'warmup_ratio' nor 'warmup_steps'; "
                "dropping warmup entirely (was ratio=%s).", ratio
            )

    if "warmup_steps" in cfg and not _accepts("warmup_steps"):
        logger.warning("target does not accept 'warmup_steps'; dropping it.")
        cfg.pop("warmup_steps")

    return cfg


def _format(row: dict) -> dict:
    from .baselines import EXTRACT_INSTRUCTION
    prompt = EXTRACT_INSTRUCTION.format(note=row["note"]).rstrip()
    # Conversational format: TRL's SFTTrainer renders this through the
    # tokenizer's chat template (with EOS/turn-end tokens handled by the
    # template itself). This is required for completion-only loss masking:
    # TRL locates the assistant span via the template's {% generation %}
    # markers, which only exist for chat-formatted (messages) datasets.
    return {"messages": [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": json.dumps(row["label"])},
    ]}


def _configure_completion_only_loss(tokenizer):
    """Pick a completion-only / assistant-only loss masking mechanism
    supported by the installed TRL version.

    Tries, in order:
    1. `SFTConfig(assistant_only_loss=True)` -- current TRL, uses the chat
       template's {% generation %} markers via return_assistant_tokens_mask.
    2. `SFTConfig(completion_only_loss=True)` -- older synonym, same
       requirement.
    3. `DataCollatorForCompletionOnlyLM` -- oldest API; matches on the
       rendered response_template string, independent of chat-template
       markers.

    Returns (extra_sft_config_kwargs, data_collator_or_None). Raises if
    none of these are available, rather than training without masking.
    """
    from trl import SFTConfig

    try:
        kwarg = _first_supported_kwarg(
            SFTConfig, ["assistant_only_loss", "completion_only_loss"])
        return {kwarg: True}, None
    except ValueError:
        pass

    try:
        from trl import DataCollatorForCompletionOnlyLM
    except ImportError as exc:
        raise RuntimeError(
            "No supported completion-only loss masking API found: SFTConfig "
            "accepts neither assistant_only_loss nor completion_only_loss, "
            "and trl.DataCollatorForCompletionOnlyLM is not importable. "
            "Refusing to train on the full sequence silently -- check the "
            "installed trl version."
        ) from exc

    collator = DataCollatorForCompletionOnlyLM(
        response_template=_ASSISTANT_RESPONSE_TEMPLATE, tokenizer=tokenizer)
    return {}, collator


def _verify_masking(trainer, tokenizer) -> None:
    """Hard runtime assertion that completion-only loss masking is
    actually active: decode one training batch's labels and confirm the
    instruction is masked out (-100) while the JSON answer is not.

    This is the check Task 11 Step 4 (docs/SERVER-RUN-CHECKLIST.md) could
    only ever specify manually, because it is hardware-blocked. Raises
    rather than proceeding if masking looks wrong, so a bad run doesn't
    burn GPU hours before anyone notices.
    """
    from .baselines import EXTRACT_INSTRUCTION

    batch = next(iter(trainer.get_train_dataloader()))
    labels = batch["labels"][0].tolist()
    masked = sum(1 for t in labels if t == -100)
    visible_ids = [t for t in labels if t != -100]
    visible_text = tokenizer.decode(visible_ids) if visible_ids else ""

    print(f"[verify_masking] {masked}/{len(labels)} label tokens masked; "
          f"visible text starts: {visible_text[:80]!r}")

    if masked == 0:
        raise RuntimeError(
            "verify_masking: no labels are masked (-100) in the first "
            "batch -- completion-only loss masking is not active, so the "
            "model would train on the full prompt. Check "
            "ensure_generation_markers() and the assistant_only_loss / "
            "completion_only_loss configuration."
        )
    if not visible_ids:
        raise RuntimeError(
            "verify_masking: every label is masked (-100) in the first "
            "batch -- there is no supervised signal at all. Check the "
            "response template / generation markers."
        )
    instruction_marker = EXTRACT_INSTRUCTION.split("\n", 1)[0]
    if instruction_marker in visible_text:
        raise RuntimeError(
            "verify_masking: instruction text is visible in the unmasked "
            f"labels ({visible_text[:200]!r}...) -- completion-only loss "
            "masking is not correctly excluding the prompt. Aborting "
            "before spending GPU hours."
        )


@app.command()
def main(config: str = "configs/train_qlora_8b.yaml", resume: bool = True) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    proc = Path("data/processed")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    ensure_generation_markers(tokenizer)

    def load(name: str) -> Dataset:
        rows = [json.loads(l) for l in (proc / f"{name}.jsonl").open(encoding="utf-8")]
        return Dataset.from_list([_format(r) for r in rows])

    train_ds, val_ds = load("train"), load("val")

    q = cfg["quantization"]
    bnb = BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=getattr(torch, q["bnb_4bit_compute_dtype"]),
    )
    peft_cfg = LoraConfig(task_type="CAUSAL_LM", **cfg["lora"])

    t = resolve_warmup_kwargs(cfg["training"], target=SFTConfig)
    loss_kwargs, fallback_collator = _configure_completion_only_loss(tokenizer)
    args = SFTConfig(
        output_dir=cfg["output_dir"], seed=cfg["seed"],
        report_to="wandb", run_name=Path(cfg["output_dir"]).name,
        save_total_limit=3, packing=False, **loss_kwargs, **t,
    )

    dtype_kwarg = resolve_dtype_kwarg(AutoModelForCausalLM.from_pretrained)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], quantization_config=bnb, device_map="auto",
        **{dtype_kwarg: getattr(torch, q["bnb_4bit_compute_dtype"])},
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_cfg,
        processing_class=tokenizer,
        data_collator=fallback_collator,
    )

    if cfg.get("verify_masking", True):
        _verify_masking(trainer, tokenizer)

    ckpts = sorted(Path(cfg["output_dir"]).glob("checkpoint-*")) if resume else []
    trainer.train(resume_from_checkpoint=bool(ckpts))
    trainer.save_model(cfg["output_dir"])


if __name__ == "__main__":
    app()
