#!/usr/bin/env python3
"""Pre-download everything Kaggle needs so GPU quota isn't wasted on data prep.

Run this on your local machine (Windows or Mac) BEFORE uploading to Kaggle.

What it does:
  1. Phase 1  — downloads NVD / OSV / GHSA / CVEfixes corpus
  2. Phase 1b — merges + tokenizes to HuggingFace arrow format
  3. Phase 3  — builds SFT instruction dataset
  4. (opt)    — pre-caches Qwen2.5-Coder-3B weights locally
  5.           — zips data/ ready for Kaggle dataset upload

Usage:
  python scripts/prepare_kaggle.py                  # phases 1 + 1b + 3 + zip
  python scripts/prepare_kaggle.py --cache-model    # also download model weights
  python scripts/prepare_kaggle.py --skip-ingest    # corpus already done, run 1b+3
  python scripts/prepare_kaggle.py --zip-only       # just re-zip existing data/
"""

import argparse
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY   = sys.executable                     # use whichever python ran this script

# ── Helpers ───────────────────────────────────────────────────────────────────

def _banner(msg: str):
    bar = "─" * 54
    print(f"\n{bar}\n  {msg}\n{bar}")


def _run(script: Path, desc: str):
    _banner(desc)
    t0 = time.time()
    result = subprocess.run([PY, str(script)], cwd=ROOT)
    elapsed = int(time.time() - t0)
    h, m, s = elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60
    if result.returncode != 0:
        print(f"\n[FAIL] {desc} — exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)
    print(f"\n[OK] {desc} — {h}:{m:02d}:{s:02d}")


def _cache_model(model_id: str):
    _banner(f"Caching model: {model_id}")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        print("Downloading tokenizer ...")
        AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        print("Downloading model weights (fp16) — this is ~6 GB ...")
        AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        print("[OK] Model cached to HuggingFace cache directory.")
        import huggingface_hub
        print(f"     Cache path: {huggingface_hub.constants.HF_HUB_CACHE}")
        print("     Upload this directory as a Kaggle dataset if you want offline training.")
    except ImportError:
        print("[SKIP] transformers not installed — run: pip install transformers torch")


def _zip_data(output_zip: Path):
    _banner(f"Zipping data/ → {output_zip.name}")
    data_dir = ROOT / "data"
    if not data_dir.exists():
        print(f"[ERROR] {data_dir} not found — run phases first.", file=sys.stderr)
        sys.exit(1)

    files = list(data_dir.rglob("*"))
    files = [f for f in files if f.is_file()]
    total_mb = sum(f.stat().st_size for f in files) / 1e6
    print(f"  {len(files)} files  ({total_mb:.0f} MB)")

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            arcname = f.relative_to(ROOT)
            zf.write(f, arcname)
            print(f"  + {arcname}")

    zip_mb = output_zip.stat().st_size / 1e6
    print(f"\n[OK] {output_zip} ({zip_mb:.0f} MB)")


def _print_upload_instructions(zip_path: Path):
    print("""
╔══════════════════════════════════════════════════════╗
║           Next steps — upload to Kaggle              ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║  1. Go to https://www.kaggle.com/datasets/new        ║
║  2. Upload:  {zip}
║  3. Dataset name: dapt-corpus                        ║
║  4. Visibility: Private                              ║
║  5. Click "Create"                                   ║
║                                                      ║
║  In your Kaggle notebook:                            ║
║    Add Data → Your Datasets → dapt-corpus            ║
║    (mounts at /kaggle/input/dapt-corpus/)            ║
║                                                      ║
║  Then set SKIP_INGEST = True in the config cell.     ║
╚══════════════════════════════════════════════════════╝
""".format(zip=zip_path.name))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Prepare Kaggle data package locally")
    p.add_argument("--skip-ingest",  action="store_true", help="Skip Phase 1 corpus download")
    p.add_argument("--cache-model",  action="store_true", help="Also download Qwen2.5-Coder-3B weights")
    p.add_argument("--zip-only",     action="store_true", help="Only re-zip existing data/")
    args = p.parse_args()

    start = time.time()
    zip_path = ROOT / "dapt_kaggle_data.zip"

    if args.zip_only:
        _zip_data(zip_path)
        _print_upload_instructions(zip_path)
        return

    if not args.skip_ingest:
        _run(ROOT / "scripts/dapt/ingest_corpus.py",      "Phase 1  — Corpus ingestion")

    _run(ROOT / "scripts/dapt/merge_and_tokenize.py",     "Phase 1b — Merge + tokenize")
    _run(ROOT / "scripts/sft/build_sft_dataset.py",       "Phase 3  — Build SFT dataset")

    if args.cache_model:
        _cache_model("Qwen/Qwen2.5-Coder-3B")

    _zip_data(zip_path)

    total = int(time.time() - start)
    h, m, s = total // 3600, (total % 3600) // 60, total % 60
    print(f"\nTotal time: {h}:{m:02d}:{s:02d}")

    _print_upload_instructions(zip_path)


if __name__ == "__main__":
    main()
