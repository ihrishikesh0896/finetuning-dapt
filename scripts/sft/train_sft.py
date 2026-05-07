#!/usr/bin/env python3
"""Phase 4: SFT fine-tuning on top of the DAPT-adapted model.

Reads:  models/dapt_adapter/           (LoRA adapter from Phase 2)
        data/sft_dataset/train.jsonl
        data/sft_dataset/eval.jsonl
Writes: models/sft_final/              (SFT LoRA adapter, best eval-loss checkpoint)
"""

import argparse
import sys
from pathlib import Path

import torch

if not torch.cuda.is_available():
    print(
        "ERROR: CUDA GPU required for training.\n"
        "  Detected backend : MPS" if torch.backends.mps.is_available() else
        "  Detected backend : CPU",
        "\n"
        "  This script targets A100 40 GB or 2×A10G.\n"
        "  bitsandbytes and bf16 training are CUDA-only.\n"
        "  Run on a cloud GPU (Colab Pro, Lambda Labs, Vast.ai, etc.).",
        file=sys.stderr,
    )
    sys.exit(1)
from datasets import load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

# ── Paths ─────────────────────────────────────────────────────────────────────

MODEL_ID     = "Qwen/Qwen2.5-Coder-3B"
DAPT_ADAPTER = Path("models/dapt_adapter")
SFT_DATA_DIR = Path("data/sft_dataset")
OUTPUT_DIR   = Path("models/sft_final")
MAX_SEQ_LEN  = 2048

# ── LoRA (SFT tier) ───────────────────────────────────────────────────────────

SFT_LORA = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
)

# ── Precision ─────────────────────────────────────────────────────────────────

def _dtype_flags() -> tuple[torch.dtype, bool, bool]:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16, True, False
    if torch.cuda.is_available():
        return torch.float16, False, True
    return torch.float32, False, False


# ── Dataset helpers ───────────────────────────────────────────────────────────

def _make_formatting_func(tokenizer):
    """Return a function that applies the Qwen2.5 chat template to each example."""
    def fmt(example: dict) -> str:
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
    return fmt


def _load_datasets():
    train = load_dataset("json", data_files=str(SFT_DATA_DIR / "train.jsonl"), split="train")
    eval_ = load_dataset("json", data_files=str(SFT_DATA_DIR / "eval.jsonl"),  split="train")
    print(f"Dataset  train={len(train)}  eval={len(eval_)}")
    return train, eval_


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Phase 4: SFT fine-tuning")
    p.add_argument("--max-steps",  type=int,  default=2_000)
    p.add_argument("--batch-size", type=int,  default=2)
    p.add_argument("--grad-accum", type=int,  default=8)
    p.add_argument("--fp16",        action="store_true", help="Force fp16 (e.g. on T4)")
    p.add_argument("--eval-steps",  type=int,  default=200)
    p.add_argument("--num-workers", type=int,  default=2,
                   help="Dataloader workers. Use 0 on Kaggle/read-only filesystems.")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = _parse_args()
    set_seed(42)
    load_dtype, use_bf16, use_fp16 = _dtype_flags()
    if args.fp16:
        load_dtype, use_bf16, use_fp16 = torch.float16, False, True
    prec = "bf16" if use_bf16 else "fp16" if use_fp16 else "fp32"
    print(f"Precision : {prec}")
    print(f"max_steps : {args.max_steps}  batch : {args.batch_size}  grad_accum : {args.grad_accum}")

    # 1. Load base model
    print(f"Loading base model: {MODEL_ID}")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=load_dtype,
        trust_remote_code=True,
        use_cache=False,
    )

    # 2. Apply DAPT adapter and bake it into the weights.
    #    This gives a clean foundation for the SFT LoRA without
    #    PEFT multi-adapter bookkeeping. Phase 5 will merge the
    #    SFT LoRA on top of this already-dapt-adapted base.
    print(f"Loading DAPT adapter: {DAPT_ADAPTER}")
    model = PeftModel.from_pretrained(base, str(DAPT_ADAPTER))
    print("Merging DAPT adapter into base weights ...")
    model = model.merge_and_unload()

    # 3. Add SFT LoRA on the merged model
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, SFT_LORA)
    model.print_trainable_parameters()

    # 4. Tokenizer — load from DAPT dir so any vocab additions are preserved
    tokenizer = AutoTokenizer.from_pretrained(str(DAPT_ADAPTER), trust_remote_code=True)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"  # left-padding breaks loss masking in SFT

    # 5. Datasets
    train_ds, eval_ds = _load_datasets()
    formatting_func = _make_formatting_func(tokenizer)

    # 6. Training
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sft_cfg = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=use_bf16,
        fp16=use_fp16,
        max_seq_length=MAX_SEQ_LEN,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=args.num_workers,
        report_to="none",
        seed=42,
        dataset_kwargs={"skip_prepare_dataset": False},
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        formatting_func=formatting_func,
    )

    print("Starting SFT training ...")
    trainer.train()

    # 7. Save the best checkpoint's adapter
    #    (load_best_model_at_end restores the best weights before we hit here)
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"SFT adapter saved → {OUTPUT_DIR}/")
    print("Note: this adapter sits on top of the DAPT-merged base.")
    print("Phase 5 (quantize.py) will merge it into full weights and export GGUF.")


if __name__ == "__main__":
    main()
