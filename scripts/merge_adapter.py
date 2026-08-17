"""Merge the LoRA adapter into base weights for single-model serving."""
import sys
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id, adapter_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16,
                                             device_map="cpu")
model = PeftModel.from_pretrained(model, adapter_dir)
model = model.merge_and_unload()
model.save_pretrained(out_dir, safe_serialization=True)
AutoTokenizer.from_pretrained(base_id).save_pretrained(out_dir)
print(f"merged -> {out_dir}")
