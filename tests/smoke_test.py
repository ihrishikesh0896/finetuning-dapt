#!/usr/bin/env python3
"""Layer 2 smoke test — no GPU, no model download, no network.

Tests corpus parsers, text formatters, SFT dataset builder, and JSON
validity of every assistant turn end-to-end on synthetic fixture data.

Run from the repo root:
    python3 tests/smoke_test.py
"""

import json
import sys
import tempfile
import textwrap
import traceback
from pathlib import Path

# ── Make repo root importable ─────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PASS = []
FAIL = []


def test(name: str):
    """Decorator that registers and runs a test case."""
    def decorator(fn):
        try:
            fn()
            PASS.append(name)
            print(f"  ✓  {name}")
        except Exception:
            FAIL.append(name)
            print(f"  ✗  {name}")
            traceback.print_exc()
        return fn
    return decorator


# ── Fixtures ──────────────────────────────────────────────────────────────────

NVD_ITEM = {
    "cve": {
        "CVE_data_meta": {"ID": "CVE-2021-44228"},
        "description": {"description_data": [{"lang": "en", "value": "Log4Shell RCE"}]},
    },
    "impact": {"baseMetricV3": {"cvssV3": {"baseScore": 10.0}}},
    "configurations": {
        "nodes": [{"cpe_match": [
            {"cpe23Uri": "cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*", "vulnerable": True}
        ]}]
    },
}

OSV_RECORD = {
    "id": "GHSA-jfh8-c2jp-1234",
    "aliases": ["CVE-2021-44228"],
    "ecosystem": "Maven",
    "summary": "Log4Shell critical RCE",
    "details": "A JNDI injection vulnerability exists in Apache Log4j 2.",
    "affected": [
        {
            "package": {"name": "log4j-core", "ecosystem": "Maven"},
            "ranges": [{"type": "ECOSYSTEM", "events": [
                {"introduced": "2.0"}, {"fixed": "2.15.0"}
            ]}],
        }
    ],
}

GHSA_RAW = {
    "id": "GHSA-jfh8-c2jp-1234",
    "summary": "Log4Shell RCE in Apache Log4j",
    "severity": [{"type": "CVSS_V3", "score": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H"}],
    "affected": [{
        "package": {"name": "log4j-core", "ecosystem": "Maven"},
        "ranges": [{"type": "ECOSYSTEM", "events": [
            {"introduced": "2.0"}, {"fixed": "2.15.0"}
        ]}],
    }],
}

CVEFIXES_RECORD = {
    "source": "cvefixes",
    "cve_id": "CVE-2021-44228",
    "cwe_id": "CWE-502",
    "pre_patch_code": textwrap.dedent("""\
        import org.apache.logging.log4j.LogManager;
        import org.apache.logging.log4j.Logger;

        public class App {
            static Logger log = LogManager.getLogger();
            public void handle(String input) {
                log.error("Received: " + input);
            }
        }
    """),
    "post_patch_code": textwrap.dedent("""\
        import org.apache.logging.log4j.LogManager;
        import org.apache.logging.log4j.Logger;

        public class App {
            static Logger log = LogManager.getLogger();
            public void handle(String input) {
                log.error("Received: {}", input);
            }
        }
    """),
    "file_path": "src/App.java",
    "programming_language": "java",
}

MANIFEST_RECORD = {
    "source": "manifest",
    "manifest_type": "requirements.txt",
    "ecosystem": "PyPI",
    "package": "requests",
    "version": "2.25.0",
    "osv_id": "GHSA-jfh8-c2jp-1234",
    "manifest_content": "requests==2.25.0\n",
    "code_snippet": textwrap.dedent("""\
        import requests

        def fetch(url, data):
            return requests.post(url, data=data)
    """),
}


# ── Tests: ingest_corpus parsers ──────────────────────────────────────────────

@test("nvd: _parse_nvd_item extracts fields correctly")
def _():
    from scripts.dapt.ingest_corpus import _parse_nvd_item
    rec = _parse_nvd_item(NVD_ITEM)
    assert rec is not None
    assert rec["cve_id"] == "CVE-2021-44228"
    assert rec["cvss_score"] == 10.0
    assert rec["description"] == "Log4Shell RCE"
    assert any("log4j" in cpe for cpe in rec["affected_versions"])


@test("nvd: _parse_nvd_item returns None on malformed input")
def _():
    from scripts.dapt.ingest_corpus import _parse_nvd_item
    assert _parse_nvd_item({}) is None
    assert _parse_nvd_item({"cve": {}}) is None


@test("osv: _parse_osv_record extracts fields correctly")
def _():
    from scripts.dapt.ingest_corpus import _parse_osv_record
    rec = _parse_osv_record(OSV_RECORD, "Maven")
    assert rec["id"] == "GHSA-jfh8-c2jp-1234"
    assert rec["ecosystem"] == "Maven"
    assert rec["summary"] == "Log4Shell critical RCE"
    assert rec["affected"][0]["package"] == "log4j-core"


@test("ghsa: _parse_ghsa_file extracts fields correctly")
def _():
    from scripts.dapt.ingest_corpus import _parse_ghsa_file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(GHSA_RAW, f)
        tmp = Path(f.name)
    rec = _parse_ghsa_file(tmp)
    tmp.unlink()
    assert rec is not None
    assert rec["ghsa_id"] == "GHSA-jfh8-c2jp-1234"
    assert rec["package"] == "log4j-core"
    assert rec["patched_version"] == "2.15.0"
    assert "CVSS" in rec["severity"]


@test("ghsa: _parse_ghsa_file returns None on bad JSON")
def _():
    from scripts.dapt.ingest_corpus import _parse_ghsa_file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("not json")
        tmp = Path(f.name)
    assert _parse_ghsa_file(tmp) is None
    tmp.unlink()


@test("manifests: manifest renderers produce non-empty output for all ecosystems")
def _():
    from scripts.dapt.ingest_corpus import _make_requirements, _make_package_json, _make_pom, _make_go_mod
    for render, pkg, ver in [
        (_make_requirements, "requests", "2.25.0"),
        (_make_package_json, "axios", "0.21.0"),
        (_make_pom, "org.apache.logging.log4j:log4j-core", "2.14.1"),
        (_make_go_mod, "golang.org/x/net", "0.0.0-20210119"),
    ]:
        manifest, snippet = render(pkg, ver)
        assert manifest.strip(), f"{render.__name__} returned empty manifest"
        assert snippet.strip(), f"{render.__name__} returned empty snippet"
        assert pkg.split(":")[-1] in manifest or pkg in manifest


# ── Tests: merge_and_tokenize formatters ──────────────────────────────────────

@test("formatter: record_to_text produces non-empty output for all source types")
def _():
    from scripts.dapt.merge_and_tokenize import record_to_text
    samples = [
        {**NVD_ITEM, "source": "nvd", "cve_id": "CVE-2021-44228",
         "description": "test", "cvss_score": 10.0, "affected_versions": ["cpe:2.3:a:a:b:1:*"]},
        {**OSV_RECORD, "source": "osv"},
        {**GHSA_RAW, "source": "ghsa", "ghsa_id": "GHSA-x", "package": "log4j",
         "ecosystem": "Maven", "vulnerable_version_range": "2.0", "patched_version": "2.15.0",
         "severity": "10.0", "summary": "test"},
        CVEFIXES_RECORD,
        MANIFEST_RECORD,
        {**MANIFEST_RECORD, "source": "manifest_synthetic"},
    ]
    for rec in samples:
        text = record_to_text(rec)
        assert text.strip(), f"Empty text for source={rec.get('source')}"


@test("formatter: fallback for unknown source type")
def _():
    from scripts.dapt.merge_and_tokenize import record_to_text
    text = record_to_text({"source": "mystery", "foo": "bar"})
    assert "bar" in text


# ── Tests: SFT dataset builder ────────────────────────────────────────────────

def _make_corpus_dir(tmp: Path):
    """Write minimal fixture JSONL files to tmp/dapt_corpus/."""
    corpus = tmp / "dapt_corpus"
    corpus.mkdir()

    (corpus / "nvd.jsonl").write_text(
        json.dumps({
            "source": "nvd",
            "cve_id": "CVE-2021-44228",
            "description": "Log4Shell",
            "cvss_score": 10.0,
            "affected_versions": ["cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*"],
        }) + "\n"
    )
    (corpus / "osv.jsonl").write_text(
        json.dumps(OSV_RECORD) + "\n"
    )
    (corpus / "ghsa.jsonl").write_text(
        json.dumps({
            "source": "ghsa",
            "ghsa_id": "GHSA-jfh8-c2jp-1234",
            "package": "log4j-core",
            "ecosystem": "Maven",
            "vulnerable_version_range": "2.0",
            "patched_version": "2.15.0",
            "severity": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H",
            "summary": "Log4Shell",
        }) + "\n"
    )
    (corpus / "cvefixes.jsonl").write_text(
        json.dumps(CVEFIXES_RECORD) + "\n"
    )
    (corpus / "manifests.jsonl").write_text(
        json.dumps(MANIFEST_RECORD) + "\n"
    )
    return corpus


@test("sft builder: load_corpus reads all five JSONL files")
def _():
    import importlib, scripts.sft.build_sft_dataset as bld

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _make_corpus_dir(tmp)

        # Patch CORPUS_DIR
        orig = bld.CORPUS_DIR
        bld.CORPUS_DIR = tmp / "dapt_corpus"
        try:
            corpus = bld.load_corpus()
        finally:
            bld.CORPUS_DIR = orig

    assert "CVE-2021-44228" in corpus["nvd_idx"]
    assert len(corpus["cvefixes"]) == 1
    assert len(corpus["manifests"]) == 1
    # OSV alias index should map CVE → record
    assert "CVE-2021-44228" in corpus["osv_cve"]


@test("sft builder: build() produces examples with valid JSON assistant turns")
def _():
    import scripts.sft.build_sft_dataset as bld

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _make_corpus_dir(tmp)

        orig_corpus = bld.CORPUS_DIR
        orig_target = bld.TARGET_TOTAL
        bld.CORPUS_DIR   = tmp / "dapt_corpus"
        bld.TARGET_TOTAL = 4      # tiny run
        try:
            corpus   = bld.load_corpus()
            examples = bld.build(corpus)
        finally:
            bld.CORPUS_DIR   = orig_corpus
            bld.TARGET_TOTAL = orig_target

    assert len(examples) == 4

    required_keys = {"package", "installed_version", "vulnerable_range", "cve_id",
                     "cvss_score", "vulnerable_attribute", "trivy_confirmed",
                     "reachability_verdict", "fix"}
    verdict_values = {"CONFIRMED_REACHABLE", "POTENTIALLY_REACHABLE", "UNREACHABLE"}

    for ex in examples:
        assert len(ex["messages"]) == 2
        assert ex["messages"][0]["role"] == "user"
        assert ex["messages"][1]["role"] == "assistant"
        # Assistant turn must be valid JSON
        out = json.loads(ex["messages"][1]["content"])
        missing = required_keys - out.keys()
        assert not missing, f"Missing keys: {missing}"
        assert out["reachability_verdict"] in verdict_values
        assert isinstance(out["cvss_score"], float)
        assert isinstance(out["trivy_confirmed"], bool)
        assert isinstance(out["fix"]["code_change"], str)


@test("sft builder: user turn contains all three XML tags")
def _():
    import scripts.sft.build_sft_dataset as bld

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _make_corpus_dir(tmp)

        orig_corpus = bld.CORPUS_DIR
        orig_target = bld.TARGET_TOTAL
        bld.CORPUS_DIR   = tmp / "dapt_corpus"
        bld.TARGET_TOTAL = 2
        try:
            corpus   = bld.load_corpus()
            examples = bld.build(corpus)
        finally:
            bld.CORPUS_DIR   = orig_corpus
            bld.TARGET_TOTAL = orig_target

    for ex in examples:
        user = ex["messages"][0]["content"]
        for tag in ("<manifest>", "<code>", "<trivy_finding>"):
            assert tag in user, f"Missing {tag} in user turn"
        assert "Analyze this finding" in user


@test("sft builder: write_splits creates train.jsonl and eval.jsonl")
def _():
    import scripts.sft.build_sft_dataset as bld

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _make_corpus_dir(tmp)

        orig_corpus = bld.CORPUS_DIR
        orig_output = bld.OUTPUT_DIR
        orig_target = bld.TARGET_TOTAL
        orig_eval   = bld.EVAL_RATIO
        bld.CORPUS_DIR   = tmp / "dapt_corpus"
        bld.OUTPUT_DIR   = tmp / "sft_dataset"
        bld.TARGET_TOTAL = 10
        bld.EVAL_RATIO   = 0.2
        try:
            corpus   = bld.load_corpus()
            examples = bld.build(corpus)
            bld.write_splits(examples)
        finally:
            bld.CORPUS_DIR   = orig_corpus
            bld.OUTPUT_DIR   = orig_output
            bld.TARGET_TOTAL = orig_target
            bld.EVAL_RATIO   = orig_eval

        train = list((tmp / "sft_dataset" / "train.jsonl").open())
        eval_ = list((tmp / "sft_dataset" / "eval.jsonl").open())
        assert len(train) == 8,  f"Expected 8 train, got {len(train)}"
        assert len(eval_) == 2,  f"Expected 2 eval,  got {len(eval_)}"
        for line in train + eval_:
            json.loads(line)   # must be valid JSON


@test("sft builder: _extract_sink finds known-dangerous call patterns")
def _():
    from scripts.sft.build_sft_dataset import _extract_sink

    code = textwrap.dedent("""\
        import pickle
        data = request.body
        obj = pickle.loads(data)
        return obj
    """)
    sink = _extract_sink(code, "python", "app/views.py")
    assert "pickle" in sink["sink_function"]
    assert sink["sink_line"] == 3
    assert sink["sink_file"] == "app/views.py"


@test("sft builder: _make_code_diff returns a non-empty unified diff")
def _():
    from scripts.sft.build_sft_dataset import _make_code_diff

    pre  = "result = eval(user_input)\n"
    post = "result = safe_eval(user_input)\n"
    diff = _make_code_diff(pre, post)
    assert "-" in diff and "+" in diff


# ── Summary ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\nRunning {len(PASS) + len(FAIL)} tests ...\n")
    total = len(PASS) + len(FAIL)
    print(f"\n{'─' * 40}")
    print(f"  Passed: {len(PASS)}/{total}")
    if FAIL:
        print(f"  Failed: {FAIL}")
        sys.exit(1)
    else:
        print("  All tests passed.")
