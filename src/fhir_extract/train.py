"""QLoRA fine-tune. Resumable by design -- server access is intermittent."""
import json
from pathlib import Path
import typer
import yaml

app = typer.Typer()


def _format(row: dict) -> str:
    from .baselines import EXTRACT_INSTRUCTION
    prompt = EXTRACT_INSTRUCTION.format(note=row["note"]).rstrip()
    return f"{prompt}\n{json.dumps(row['label'])}"


@app.command()
def main(config: str = "configs/train_qlora_8b.yaml", resume: bool = True) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    proc = Path("data/processed")

    def load(name: str) -> Dataset:
        rows = [json.loads(l) for l in (proc / f"{name}.jsonl").open(encoding="utf-8")]
        return Dataset.from_dict({"text": [_format(r) for r in rows]})

    train_ds, val_ds = load("train"), load("val")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    q = cfg["quantization"]
    bnb = BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=getattr(torch, q["bnb_4bit_compute_dtype"]),
    )
    peft_cfg = LoraConfig(task_type="CAUSAL_LM", **cfg["lora"])
    t = cfg["training"]
    args = SFTConfig(
        output_dir=cfg["output_dir"], seed=cfg["seed"],
        report_to="wandb", run_name=Path(cfg["output_dir"]).name,
        save_total_limit=3, packing=False, **t,
    )

    trainer = SFTTrainer(
        model=cfg["model_id"],
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_cfg,
        processing_class=tokenizer,
        model_init_kwargs={"quantization_config": bnb, "device_map": "auto"},
    )

    ckpts = sorted(Path(cfg["output_dir"]).glob("checkpoint-*")) if resume else []
    trainer.train(resume_from_checkpoint=bool(ckpts))
    trainer.save_model(cfg["output_dir"])


if __name__ == "__main__":
    app()
