#!/usr/bin/env python3
"""Phase 1b: Merge corpus JSONL → plain text → tokenized HF arrow dataset.

Reads:  data/dapt_corpus/*.jsonl
Writes: data/dapt_corpus/merged.jsonl
        data/dapt_tokenized/  (HuggingFace arrow format)
"""

import json
from pathlib import Path

# Heavy deps imported lazily so pure-Python formatters stay importable without the ML stack.

CORPUS_DIR = Path("data/dapt_corpus")
MERGED_PATH = CORPUS_DIR / "merged.jsonl"
OUTPUT_DIR = Path("data/dapt_tokenized")
MODEL_ID = "Qwen/Qwen2.5-Coder-3B"
MAX_LENGTH = 2048


# ── Plain-text formatters (no chat template — raw causal LM) ─────────────────

def _fmt_nvd(r: dict) -> str:
    lines = [f"CVE: {r.get('cve_id', '')}"]
    if r.get("cvss_score") is not None:
        lines.append(f"CVSS Score: {r['cvss_score']}")
    if r.get("description"):
        lines.append(f"Description: {r['description']}")
    if r.get("affected_versions"):
        lines.append("Affected CPEs:\n" + "\n".join(r["affected_versions"][:5]))
    return "\n".join(lines)


def _fmt_osv(r: dict) -> str:
    lines = [f"Advisory: {r.get('id', '')} ({r.get('ecosystem', '')})"]
    if r.get("summary"):
        lines.append(f"Summary: {r['summary']}")
    if r.get("details"):
        lines.append(f"Details: {r['details'][:1000]}")
    for aff in r.get("affected", [])[:3]:
        pkg = aff.get("package", "")
        if pkg:
            lines.append(f"Package: {pkg}")
    return "\n".join(lines)


def _fmt_ghsa(r: dict) -> str:
    lines = [f"Advisory: {r.get('ghsa_id', '')}"]
    if r.get("package"):
        lines.append(f"Package: {r['package']} ({r.get('ecosystem', '')})")
    if r.get("vulnerable_version_range"):
        lines.append(f"Vulnerable from: {r['vulnerable_version_range']}")
    if r.get("patched_version"):
        lines.append(f"Fixed in: {r['patched_version']}")
    if r.get("severity"):
        lines.append(f"Severity: {r['severity']}")
    if r.get("summary"):
        lines.append(f"Summary: {r['summary']}")
    return "\n".join(lines)


def _fmt_cvefixes(r: dict) -> str:
    lines = [f"CVE: {r.get('cve_id', '')}"]
    if r.get("cwe_id"):
        lines.append(f"CWE: {r['cwe_id']}")
    lang = r.get("programming_language", "").lower()
    if r.get("pre_patch_code"):
        lines.append(f"Vulnerable code:\n```{lang}\n{r['pre_patch_code'][:800]}\n```")
    if r.get("post_patch_code"):
        lines.append(f"Fixed code:\n```{lang}\n{r['post_patch_code'][:800]}\n```")
    if r.get("file_path"):
        lines.append(f"File: {r['file_path']}")
    return "\n".join(lines)


def _fmt_manifest(r: dict) -> str:
    lines = []
    if r.get("osv_id"):
        lines.append(f"Advisory: {r['osv_id']}")
    lines.append(f"Manifest ({r.get('manifest_type', '')}):")
    lines.append(r.get("manifest_content", "").strip())
    lines.append("Code:")
    lines.append(r.get("code_snippet", "").strip())
    return "\n".join(lines)


_FORMATTERS = {
    "nvd": _fmt_nvd,
    "osv": _fmt_osv,
    "ghsa": _fmt_ghsa,
    "cvefixes": _fmt_cvefixes,
    "manifest": _fmt_manifest,
    "manifest_synthetic": _fmt_manifest,
}


def record_to_text(rec: dict) -> str:
    fmt = _FORMATTERS.get(rec.get("source", ""))
    if fmt:
        return fmt(rec)
    return "\n".join(f"{k}: {v}" for k, v in rec.items() if isinstance(v, str) and v)


# ── Merge ─────────────────────────────────────────────────────────────────────

def merge() -> int:
    files = sorted(f for f in CORPUS_DIR.glob("*.jsonl") if f.name != "merged.jsonl")
    print(f"Merging {len(files)} files: {[f.name for f in files]}")
    count = 0
    with MERGED_PATH.open("w") as out:
        for f in files:
            with f.open() as inp:
                for line in inp:
                    line = line.strip()
                    if line:
                        out.write(line + "\n")
                        count += 1
    print(f"Merged {count} records → {MERGED_PATH}")
    return count


# ── Tokenize ──────────────────────────────────────────────────────────────────

def tokenize():
    from datasets import Dataset  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415
    from tqdm import tqdm  # noqa: PLC0415
    print(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    texts = []
    print("Formatting records as plain text ...")
    with MERGED_PATH.open() as f:
        for line in tqdm(f, desc="format"):
            try:
                rec = json.loads(line)
                text = record_to_text(rec)
                if text.strip():
                    texts.append(text)
            except json.JSONDecodeError:
                pass
    print(f"  {len(texts)} non-empty documents")

    def _tokenize(batch: dict) -> dict:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )

    ds = Dataset.from_dict({"text": texts})
    ds = ds.map(_tokenize, batched=True, batch_size=1000, remove_columns=["text"], desc="tokenize")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(OUTPUT_DIR))
    print(f"Saved {len(ds)} tokenized examples → {OUTPUT_DIR}/")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    merge()
    tokenize()
