import argparse
import os
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-dir", default="outputs/mts_sft_data")
    ap.add_argument("--out", default="outputs/mts_lora_4b")
    ap.add_argument("--base", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.base, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", 
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True
        )
    )
    model.config.use_cache = False

    ds = load_dataset("json", data_files={"train": f"{args.sft_dir}/train.jsonl", "val": f"{args.sft_dir}/val.jsonl"})
    
    cfg = SFTConfig(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=2, gradient_accumulation_steps=8,
        learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=10, bf16=True, gradient_checkpointing=True,
        max_length=4096, completion_only_loss=True,
        eval_strategy="steps", eval_steps=50,
        save_strategy="steps", save_steps=50, save_total_limit=2,
        report_to="none", seed=42
    )
    
    peft = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")
    
    trainer = SFTTrainer(
        model=model, args=cfg, train_dataset=ds["train"], 
        eval_dataset=ds["val"], peft_config=peft, processing_class=tok
    )
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)

if __name__ == "__main__":
    main()