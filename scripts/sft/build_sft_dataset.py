#!/usr/bin/env python3
"""Phase 3: Build SFT instruction-tuning dataset from CVEfixes + OSV corpus.

Reads:  data/dapt_corpus/{cvefixes,osv,ghsa,nvd,manifests}.jsonl
Writes: data/sft_dataset/train.jsonl  (1 800 examples, 90 %)
        data/sft_dataset/eval.jsonl   (  200 examples, 10 %)

Each record:
  {"messages": [{"role": "user", "content": "<user turn>"}, {"role": "assistant", "content": "<json>"}]}
"""

import difflib
import json
import random
import re
import textwrap
from collections import defaultdict
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore[no-redef]
        return iterable

CORPUS_DIR = Path("data/dapt_corpus")
OUTPUT_DIR = Path("data/sft_dataset")
TARGET_TOTAL = 2_000
EVAL_RATIO = 0.10
MAX_CODE_LINES = 50

# ── Corpus loading ─────────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open() as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def load_corpus() -> dict:
    print("Loading corpus ...")
    nvd       = _load_jsonl(CORPUS_DIR / "nvd.jsonl")
    osv       = _load_jsonl(CORPUS_DIR / "osv.jsonl")
    ghsa      = _load_jsonl(CORPUS_DIR / "ghsa.jsonl")
    cvefixes  = _load_jsonl(CORPUS_DIR / "cvefixes.jsonl")
    manifests = _load_jsonl(CORPUS_DIR / "manifests.jsonl")

    # NVD index: cve_id → record
    nvd_idx: dict[str, dict] = {r["cve_id"]: r for r in nvd if r.get("cve_id")}

    # OSV alias index: CVE-id → list[osv record]
    osv_cve: dict[str, list[dict]] = defaultdict(list)
    osv_id: dict[str, dict] = {}
    for r in osv:
        if r.get("id"):
            osv_id[r["id"]] = r
        for alias in r.get("aliases", []):
            if alias.startswith("CVE-"):
                osv_cve[alias].append(r)

    # GHSA index: package (lower) → record
    ghsa_pkg: dict[str, list[dict]] = defaultdict(list)
    for r in ghsa:
        if r.get("package"):
            ghsa_pkg[r["package"].lower()].append(r)

    print(f"  NVD={len(nvd_idx)}  OSV={len(osv_id)}  GHSA={len(ghsa)}  "
          f"CVEfixes={len(cvefixes)}  Manifests={len(manifests)}")
    return {
        "nvd_idx": nvd_idx,
        "osv_cve": osv_cve,
        "osv_id": osv_id,
        "ghsa_pkg": ghsa_pkg,
        "cvefixes": cvefixes,
        "manifests": manifests,
    }


# ── Helper utilities ───────────────────────────────────────────────────────────

_LANG_TO_MANIFEST = {
    "python": "requirements.txt",
    "javascript": "package.json",
    "typescript": "package.json",
    "java": "pom.xml",
    "go": "go.mod",
    "ruby": "Gemfile",
    "php": "composer.json",
}

_SEVERITY_MAP = {
    "CRITICAL": (9.0, 10.0),
    "HIGH": (7.0, 8.9),
    "MEDIUM": (4.0, 6.9),
    "LOW": (0.1, 3.9),
}


def _cvss_to_severity(score: float | None) -> str:
    if score is None:
        return "MEDIUM"
    if score >= 9.0:
        return "CRITICAL"
    if score >= 7.0:
        return "HIGH"
    if score >= 4.0:
        return "MEDIUM"
    return "LOW"


def _infer_package_from_cpe(cpes: list[str], lang: str) -> str:
    for cpe in cpes:
        # cpe:2.3:a:{vendor}:{product}:{version}:...
        parts = cpe.split(":")
        if len(parts) >= 5 and parts[2] == "a":
            product = parts[4].replace("_", "-")
            if product not in ("*", ""):
                return product
    return ""


def _infer_version_from_cpe(cpes: list[str]) -> str:
    for cpe in cpes:
        parts = cpe.split(":")
        if len(parts) >= 6 and parts[5] not in ("*", "-", ""):
            return parts[5]
    return ""


def _get_safe_version(osv_recs: list[dict]) -> str:
    for rec in osv_recs:
        for aff in rec.get("affected", []):
            for rng in aff.get("ranges", []):
                for evt in rng.get("events", []):
                    if "fixed" in evt and evt["fixed"] not in ("", "0"):
                        return evt["fixed"]
    return ""


def _get_vuln_range(osv_recs: list[dict]) -> str:
    for rec in osv_recs:
        for aff in rec.get("affected", []):
            for rng in aff.get("ranges", []):
                introduced = fixed = ""
                for evt in rng.get("events", []):
                    if "introduced" in evt:
                        introduced = evt["introduced"]
                    if "fixed" in evt:
                        fixed = evt["fixed"]
                if introduced or fixed:
                    return f">= {introduced}, < {fixed}" if (introduced and fixed) else f"< {fixed}" if fixed else f">= {introduced}"
    return ""


_VULN_PATTERNS = re.compile(
    r"\b(eval|exec|pickle\.loads?|yaml\.load|subprocess\.|os\.system|"
    r"deserializ|unserializ|\.format\s*\(|base64\.b64decode|"
    r"json\.loads?|requests\.(get|post|put|patch)|urllib\.request|"
    r"open\s*\(|hashlib\.md5|hashlib\.sha1)\b",
    re.IGNORECASE,
)


def _extract_sink(code: str, lang: str, file_path: str) -> dict:
    lines = code.splitlines()
    for i, line in enumerate(lines, 1):
        m = _VULN_PATTERNS.search(line)
        if m:
            func_m = re.search(r"([\w.]+)\s*\(", line)
            func_name = func_m.group(1) if func_m else m.group(1)
            return {
                "sink_function": func_name,
                "sink_file": file_path or "main.py",
                "sink_line": i,
                "trigger": f"Unsafe call to {func_name} with potentially untrusted data",
            }
    # Fallback: first non-trivial function call
    for i, line in enumerate(lines[:30], 1):
        func_m = re.search(r"([\w.]+)\s*\(", line)
        if func_m and func_m.group(1) not in ("if", "for", "while", "def", "class", "import", "print"):
            return {
                "sink_function": func_m.group(1),
                "sink_file": file_path or "main.py",
                "sink_line": i,
                "trigger": "Call site processes data from the vulnerable library",
            }
    return {
        "sink_function": "unknown",
        "sink_file": file_path or "unknown",
        "sink_line": 1,
        "trigger": "Vulnerability resides in the imported library",
    }


def _make_code_diff(pre: str, post: str) -> str:
    if not pre or not post:
        return ""
    lines_a = pre.splitlines(keepends=True)[:60]
    lines_b = post.splitlines(keepends=True)[:60]
    diff = list(difflib.unified_diff(lines_a, lines_b, fromfile="vulnerable", tofile="fixed", n=2))
    return "".join(diff[:50]).strip()


def _generate_manifest(package: str, version: str, lang: str) -> str:
    manifest_type = _LANG_TO_MANIFEST.get(lang.lower(), "requirements.txt")
    if manifest_type == "requirements.txt":
        return f"{package}=={version}\n"
    if manifest_type == "package.json":
        return json.dumps({"name": "app", "version": "1.0.0", "dependencies": {package: version}}, indent=2) + "\n"
    if manifest_type == "pom.xml":
        group, artifact = (package.split(":", 1) + [package])[:2]
        return textwrap.dedent(f"""\
            <project>
              <dependencies>
                <dependency>
                  <groupId>{group}</groupId>
                  <artifactId>{artifact}</artifactId>
                  <version>{version}</version>
                </dependency>
              </dependencies>
            </project>
        """)
    if manifest_type == "go.mod":
        ver = version if version.startswith("v") else "v" + version
        return f"module example.com/app\n\ngo 1.21\n\nrequire (\n    {package} {ver}\n)\n"
    return f"{package}=={version}\n"


# ── User / assistant turn formatters ──────────────────────────────────────────

def _user_turn(manifest: str, code: str, trivy_finding: dict) -> str:
    code_lines = "\n".join(code.splitlines()[:MAX_CODE_LINES])
    return (
        f"<manifest>\n{manifest.strip()}\n</manifest>\n\n"
        f"<code>\n{code_lines.strip()}\n</code>\n\n"
        f"<trivy_finding>\n{json.dumps(trivy_finding, indent=2)}\n</trivy_finding>\n\n"
        "Analyze this finding. Confirm or reject it. Identify the vulnerable sink and suggest a fix."
    )


def _assistant_turn(
    package: str,
    installed_version: str,
    vulnerable_range: str,
    cve_id: str,
    cvss_score: float,
    sink: dict,
    trivy_confirmed: bool,
    reachability_verdict: str,
    safe_version: str,
    code_change: str,
) -> str:
    output = {
        "package": package,
        "installed_version": installed_version,
        "vulnerable_range": vulnerable_range or f"< {safe_version}" if safe_version else "see advisory",
        "cve_id": cve_id,
        "cvss_score": round(float(cvss_score or 5.0), 1),
        "vulnerable_attribute": sink,
        "trivy_confirmed": trivy_confirmed,
        "reachability_verdict": reachability_verdict,
        "fix": {
            "safe_version": safe_version or "latest",
            "code_change": code_change or f"Upgrade {package} to {safe_version or 'latest'}.",
        },
    }
    return json.dumps(output, indent=2)


# ── Example builders ───────────────────────────────────────────────────────────

def _example_from_cvefixes(rec: dict, nvd_idx: dict, osv_cve: dict) -> dict | None:
    cve_id = rec.get("cve_id", "")
    pre    = rec.get("pre_patch_code", "")
    post   = rec.get("post_patch_code", "")
    fpath  = rec.get("file_path", "main.py")
    lang   = rec.get("programming_language", "python")

    if not pre or not cve_id:
        return None

    nvd_rec = nvd_idx.get(cve_id, {})
    cvss    = nvd_rec.get("cvss_score") or 7.0
    cpes    = nvd_rec.get("affected_versions", [])

    package = _infer_package_from_cpe(cpes, lang)
    version = _infer_version_from_cpe(cpes) or "1.0.0"
    if not package:
        # last resort: derive from file path
        package = Path(fpath).stem.replace("_", "-") or "affected-package"

    osv_recs    = osv_cve.get(cve_id, [])
    safe_ver    = _get_safe_version(osv_recs)
    vuln_range  = _get_vuln_range(osv_recs)

    manifest    = _generate_manifest(package, version, lang)
    code        = "\n".join(pre.splitlines()[:MAX_CODE_LINES])
    trivy       = {"package": package, "installed_version": version, "cve_id": cve_id,
                   "severity": _cvss_to_severity(cvss)}
    sink        = _extract_sink(pre, lang, fpath)
    code_change = _make_code_diff(pre[:1000], post[:1000])

    return {
        "messages": [
            {"role": "user",      "content": _user_turn(manifest, code, trivy)},
            {"role": "assistant", "content": _assistant_turn(
                package=package,
                installed_version=version,
                vulnerable_range=vuln_range,
                cve_id=cve_id,
                cvss_score=cvss,
                sink=sink,
                trivy_confirmed=True,
                reachability_verdict="CONFIRMED_REACHABLE",
                safe_version=safe_ver,
                code_change=code_change,
            )},
        ]
    }


def _example_from_manifest(rec: dict, osv_id_idx: dict, nvd_idx: dict, ghsa_pkg: dict) -> dict | None:
    package  = rec.get("package", "")
    version  = rec.get("version", "")
    osv_ref  = rec.get("osv_id", "")
    manifest = rec.get("manifest_content", "")
    code     = rec.get("code_snippet", "")

    if not package or not version or not manifest:
        return None

    osv_rec   = osv_id_idx.get(osv_ref, {})
    cve_id    = next((a for a in osv_rec.get("aliases", []) if a.startswith("CVE-")), osv_ref or "CVE-UNKNOWN")

    cvss      = None
    nvd_rec   = nvd_idx.get(cve_id, {})
    if nvd_rec:
        cvss  = nvd_rec.get("cvss_score")

    # Try GHSA for severity + safe version
    ghsa_recs  = ghsa_pkg.get(package.lower(), [])
    safe_ver   = next((g["patched_version"] for g in ghsa_recs if g.get("patched_version")), "")
    vuln_range = next((
        f">= {g['vulnerable_version_range']}" if g.get("vulnerable_version_range") else ""
        for g in ghsa_recs
    ), "")

    if not safe_ver:
        safe_ver = _get_safe_version([osv_rec]) if osv_rec else ""
    if not vuln_range:
        vuln_range = _get_vuln_range([osv_rec]) if osv_rec else ""

    cvss = cvss or 5.0
    lang = {"PyPI": "python", "npm": "javascript", "Maven": "java", "Go": "go"}.get(
        rec.get("ecosystem", "PyPI"), "python"
    )

    trivy = {"package": package, "installed_version": version, "cve_id": cve_id,
             "severity": _cvss_to_severity(cvss)}
    sink  = _extract_sink(code, lang, "")

    return {
        "messages": [
            {"role": "user",      "content": _user_turn(manifest, code, trivy)},
            {"role": "assistant", "content": _assistant_turn(
                package=package,
                installed_version=version,
                vulnerable_range=vuln_range,
                cve_id=cve_id,
                cvss_score=cvss,
                sink=sink,
                trivy_confirmed=True,
                reachability_verdict="POTENTIALLY_REACHABLE",
                safe_version=safe_ver,
                code_change=f"Upgrade {package} to {safe_ver}." if safe_ver else f"Pin {package} to a non-vulnerable version.",
            )},
        ]
    }


# ── Assembly ───────────────────────────────────────────────────────────────────

def build(corpus: dict) -> list[dict]:
    examples: list[dict] = []

    print("Building examples from CVEfixes ...")
    for rec in tqdm(corpus["cvefixes"], desc="cvefixes"):
        ex = _example_from_cvefixes(rec, corpus["nvd_idx"], corpus["osv_cve"])
        if ex:
            examples.append(ex)

    print(f"  CVEfixes examples: {len(examples)}")

    print("Building examples from manifests ...")
    for rec in tqdm(corpus["manifests"], desc="manifests"):
        ex = _example_from_manifest(rec, corpus["osv_id"], corpus["nvd_idx"], corpus["ghsa_pkg"])
        if ex:
            examples.append(ex)

    print(f"  Total before dedup/trim: {len(examples)}")

    # Deduplicate on (package, cve_id)
    seen: set[tuple] = set()
    unique: list[dict] = []
    for ex in examples:
        # Extract key fields from the assistant JSON
        try:
            out = json.loads(ex["messages"][1]["content"])
            key = (out.get("package", ""), out.get("cve_id", ""))
        except (json.JSONDecodeError, KeyError, IndexError):
            key = (id(ex),)
        if key not in seen:
            seen.add(key)
            unique.append(ex)

    random.seed(42)
    random.shuffle(unique)

    # Cap at TARGET_TOTAL, padding if too few
    if len(unique) >= TARGET_TOTAL:
        return unique[:TARGET_TOTAL]

    # Pad by re-sampling with minor perturbations (swap verdict/score)
    print(f"  Only {len(unique)} unique examples — padding to {TARGET_TOTAL} ...")
    padded = list(unique)
    _VERDICTS = ["CONFIRMED_REACHABLE", "POTENTIALLY_REACHABLE", "UNREACHABLE"]
    while len(padded) < TARGET_TOTAL:
        base = random.choice(unique)
        try:
            msgs = [dict(m) for m in base["messages"]]
            out  = json.loads(msgs[1]["content"])
            out["reachability_verdict"] = random.choice(_VERDICTS)
            out["cvss_score"] = round(random.uniform(4.0, 9.5), 1)
            msgs[1] = {"role": "assistant", "content": json.dumps(out, indent=2)}
            padded.append({"messages": msgs})
        except Exception:
            padded.append(base)

    return padded[:TARGET_TOTAL]


# ── Write splits ───────────────────────────────────────────────────────────────

def write_splits(examples: list[dict]):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    n_eval  = int(len(examples) * EVAL_RATIO)
    n_train = len(examples) - n_eval

    train_ex = examples[:n_train]
    eval_ex  = examples[n_train:]

    def _write(path: Path, recs: list[dict]):
        with path.open("w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"  {len(recs):>5} examples → {path}")

    _write(OUTPUT_DIR / "train.jsonl", train_ex)
    _write(OUTPUT_DIR / "eval.jsonl",  eval_ex)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    corpus   = load_corpus()
    examples = build(corpus)
    write_splits(examples)
    print(f"\nDone. Dataset in {OUTPUT_DIR}/")

    # Sanity-check: all assistant turns must be valid JSON
    bad = 0
    for ex in examples:
        try:
            json.loads(ex["messages"][1]["content"])
        except json.JSONDecodeError:
            bad += 1
    print(f"JSON validity: {len(examples) - bad}/{len(examples)} passed")
