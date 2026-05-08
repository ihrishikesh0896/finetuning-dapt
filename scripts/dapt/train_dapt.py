#!/usr/bin/env python3
"""Phase 2: DAPT continued pre-training on Qwen2.5-Coder-3B with LoRA.

Reads:  data/dapt_tokenized/   (HF arrow dataset from merge_and_tokenize.py)
Writes: models/dapt_checkpoints/  (every 1 000 steps)
        models/dapt_adapter/       (final LoRA adapter)
"""

import argparse
import os
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
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

# ── Paths ─────────────────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen2.5-Coder-3B"
CHECKPOINT_DIR = Path("models/dapt_checkpoints")
ADAPTER_DIR = Path("models/dapt_adapter")

# ── LoRA ──────────────────────────────────────────────────────────────────────

LORA_CFG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
)

# ── Precision ─────────────────────────────────────────────────────────────────

def _dtype_and_flags() -> tuple[torch.dtype, bool, bool]:
    """Return (load_dtype, bf16_flag, fp16_flag) based on hardware."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16, True, False
    if torch.cuda.is_available():
        return torch.float16, False, True
    return torch.float32, False, False


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Phase 2: DAPT continued pre-training")
    p.add_argument("--max-steps",  type=int,   default=10_000)
    p.add_argument("--batch-size", type=int,   default=4)
    p.add_argument("--grad-accum", type=int,   default=8)
    p.add_argument("--fp16",        action="store_true", help="Force fp16 (e.g. on T4)")
    p.add_argument("--save-steps",  type=int,   default=1_000)
    p.add_argument("--num-workers", type=int,   default=2,
                   help="Dataloader workers. Use 0 on Kaggle/read-only filesystems.")
    p.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to tokenized HF dataset. Defaults to data/dapt_tokenized. "
             "Must be a writable path — not a symlink to /kaggle/input/.",
    )
    p.add_argument(
        "--max-train-length",
        type=int,
        default=None,
        help="Drop sequences longer than this many tokens before training. "
             "Use 512 on T4 to avoid padding waste from short CVE text padded to 2048.",
    )
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = _parse_args()
    set_seed(42)

    load_dtype, use_bf16, use_fp16 = _dtype_and_flags()
    if args.fp16:                          # explicit override for T4 / older GPUs
        load_dtype, use_bf16, use_fp16 = torch.float16, False, True
    print(f"Precision : {'bf16' if use_bf16 else 'fp16' if use_fp16 else 'fp32'}")
    print(f"max_steps : {args.max_steps}  batch : {args.batch_size}  grad_accum : {args.grad_accum}  max_train_length : {args.max_train_length or 'none'}")

    # Resolve data directory — prefer explicit arg, then env var, then default
    data_dir = Path(
        args.data_dir
        or os.environ.get("DAPT_DATA_DIR", "data/dapt_tokenized")
    )
    if not data_dir.exists():
        print(f"ERROR: tokenized dataset not found at {data_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Loading tokenized dataset from {data_dir}")

    # Model
    print(f"Loading base model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=load_dtype,
        trust_remote_code=True,
        use_cache=False,
    )
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, LORA_CFG)
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    ds = load_from_disk(str(data_dir))

    if args.max_train_length is not None:
        before = len(ds)
        ds = ds.filter(
            lambda ex: len(ex["input_ids"]) <= args.max_train_length,
            num_proc=1,
            keep_in_memory=True,   # both DDP ranks run this; skip disk cache to avoid file-lock deadlock
        )
        print(f"  Filtered to max_train_length={args.max_train_length}: {before} → {len(ds)} sequences")

    split = ds.train_test_split(test_size=0.05, seed=42)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"  train={len(train_ds)}  eval={len(eval_ds)}")

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    train_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=100,
        save_steps=args.save_steps,
        save_total_limit=5,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        load_best_model_at_end=False,
        report_to="none",
        dataloader_num_workers=args.num_workers,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        processing_class=tokenizer
    )

    print("Starting DAPT training ...")
    trainer.train()

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))
    print(f"LoRA adapter saved → {ADAPTER_DIR}/")


if __name__ == "__main__":
    main()