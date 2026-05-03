#!/usr/bin/env python3
"""Phase 1: Pull and format the DAPT pre-training corpus.

Outputs one JSONL per source to data/dapt_corpus/:
  nvd.jsonl, osv.jsonl, ghsa.jsonl, cvefixes.jsonl, manifests.jsonl
"""

import gzip
import io
import json
import random
import subprocess
import textwrap
import zipfile
from pathlib import Path

# Heavy deps imported lazily inside the functions that need them so that
# pure-Python helpers remain importable without the full ML stack installed.

OUTPUT_DIR = Path("data/dapt_corpus")
GHSA_REPO_DIR = Path("data/_ghsa_repo")

NVD_BASE = "https://nvd.nist.gov/feeds/json/cve/1.1"
NVD_YEARS = list(range(2018, 2025)) + ["recent"]

OSV_BASE = "https://storage.googleapis.com/osv-vulnerabilities"
OSV_ECOSYSTEMS = ["PyPI", "npm", "Maven", "Go"]

GHSA_REPO = "https://github.com/github/advisory-database"


def _write_jsonl(path: Path, records: list[dict]) -> int:
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return len(records)


# ── 1. NVD ────────────────────────────────────────────────────────────────────

def _parse_nvd_item(item: dict) -> dict | None:
    try:
        cve_id = item["cve"]["CVE_data_meta"]["ID"]
        descs = item["cve"]["description"]["description_data"]
        description = next((d["value"] for d in descs if d.get("lang") == "en"), "")
        impact = item.get("impact", {})
        cvss_score = (
            impact.get("baseMetricV3", {}).get("cvssV3", {}).get("baseScore")
            or impact.get("baseMetricV2", {}).get("cvssV2", {}).get("baseScore")
        )
        cpes = [
            m["cpe23Uri"]
            for node in item.get("configurations", {}).get("nodes", [])
            for m in node.get("cpe_match", [])
            if m.get("vulnerable")
        ]
        return {
            "source": "nvd",
            "cve_id": cve_id,
            "description": description,
            "cvss_score": cvss_score,
            "affected_versions": cpes[:20],
        }
    except (KeyError, StopIteration):
        return None


def ingest_nvd():
    import requests  # noqa: PLC0415
    out = OUTPUT_DIR / "nvd.jsonl"
    print(f"[NVD] → {out}")
    records = []
    for year in NVD_YEARS:
        url = f"{NVD_BASE}/nvdcve-1.1-{year}.json.gz"
        print(f"  fetching {url}")
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            with gzip.open(io.BytesIO(resp.content)) as gz:
                feed = json.load(gz)
            for item in feed.get("CVE_Items", []):
                rec = _parse_nvd_item(item)
                if rec:
                    records.append(rec)
        except Exception as exc:
            print(f"  [WARN] {year}: {exc}")
    n = _write_jsonl(out, records)
    print(f"[NVD] {n} records written")


# ── 2. OSV ────────────────────────────────────────────────────────────────────

def _parse_osv_record(raw: dict, ecosystem: str) -> dict | None:
    try:
        affected = [
            {
                "package": a.get("package", {}).get("name", ""),
                "ranges": a.get("ranges", []),
            }
            for a in raw.get("affected", [])
        ]
        return {
            "source": "osv",
            "ecosystem": ecosystem,
            "id": raw["id"],
            "summary": raw.get("summary", ""),
            "details": raw.get("details", ""),
            "affected": affected,
        }
    except KeyError:
        return None


def ingest_osv():
    import requests  # noqa: PLC0415
    from tqdm import tqdm  # noqa: PLC0415
    out = OUTPUT_DIR / "osv.jsonl"
    print(f"[OSV] → {out}")
    records = []
    for eco in OSV_ECOSYSTEMS:
        url = f"{OSV_BASE}/{eco}/all.zip"
        print(f"  fetching {url}")
        try:
            resp = requests.get(url, timeout=300)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                for name in tqdm(zf.namelist(), desc=eco, leave=False):
                    if not name.endswith(".json"):
                        continue
                    try:
                        raw = json.loads(zf.read(name))
                        rec = _parse_osv_record(raw, eco)
                        if rec:
                            records.append(rec)
                    except Exception:
                        pass
        except Exception as exc:
            print(f"  [WARN] {eco}: {exc}")
    n = _write_jsonl(out, records)
    print(f"[OSV] {n} records written")


# ── 3. GHSA ───────────────────────────────────────────────────────────────────

def _parse_ghsa_file(path: Path) -> dict | None:
    try:
        raw = json.loads(path.read_text())
        affected = raw.get("affected", [])
        pkg = affected[0].get("package", {}) if affected else {}
        ranges = affected[0].get("ranges", []) if affected else []

        patched = vuln_range = None
        for r in ranges:
            for evt in r.get("events", []):
                if "introduced" in evt:
                    vuln_range = evt["introduced"]
                if "fixed" in evt:
                    patched = evt["fixed"]

        severity = next(
            (s.get("score", "") for s in raw.get("severity", []) if "CVSS" in s.get("type", "")),
            "",
        )
        return {
            "source": "ghsa",
            "ghsa_id": raw["id"],
            "package": pkg.get("name", ""),
            "ecosystem": pkg.get("ecosystem", ""),
            "vulnerable_version_range": vuln_range,
            "patched_version": patched,
            "severity": severity,
            "summary": raw.get("summary", ""),
        }
    except (KeyError, json.JSONDecodeError):
        return None


def ingest_ghsa():
    from tqdm import tqdm  # noqa: PLC0415
    out = OUTPUT_DIR / "ghsa.jsonl"
    print(f"[GHSA] → {out}")

    if not GHSA_REPO_DIR.exists():
        print(f"  cloning {GHSA_REPO} (shallow) ...")
        subprocess.run(
            ["git", "clone", "--depth=1", GHSA_REPO, str(GHSA_REPO_DIR)],
            check=True,
        )
    else:
        print(f"  using cached clone at {GHSA_REPO_DIR}")

    advisory_root = GHSA_REPO_DIR / "advisories" / "github-reviewed"
    json_files = list(advisory_root.rglob("*.json"))
    print(f"  parsing {len(json_files)} advisories")

    records = [r for p in tqdm(json_files, desc="GHSA") if (r := _parse_ghsa_file(p))]
    n = _write_jsonl(out, records)
    print(f"[GHSA] {n} records written")


# ── 4. CVEfixes ───────────────────────────────────────────────────────────────

def ingest_cvefixes():
    from datasets import load_dataset  # noqa: PLC0415
    from tqdm import tqdm  # noqa: PLC0415
    out = OUTPUT_DIR / "cvefixes.jsonl"
    print(f"[CVEfixes] → {out}")
    try:
        ds = load_dataset("SirWang/cvefixes", split="train", trust_remote_code=True)
    except Exception as exc:
        print(f"  [WARN] could not load SirWang/cvefixes: {exc}")
        print("  Zenodo fallback requires manual download to data/_cvefixes/")
        _write_jsonl(out, [])
        return

    def _map(row: dict) -> dict:
        return {
            "source": "cvefixes",
            "cve_id": row.get("cve_id", ""),
            "cwe_id": row.get("cwe_id", ""),
            "pre_patch_code": row.get("before_fix_code", row.get("pre_patch_code", "")),
            "post_patch_code": row.get("after_fix_code", row.get("post_patch_code", "")),
            "file_path": row.get("file_path", ""),
            "programming_language": row.get("lang", row.get("programming_language", "")),
        }

    records = [_map(r) for r in tqdm(ds, desc="CVEfixes")]
    n = _write_jsonl(out, records)
    print(f"[CVEfixes] {n} records written")


# ── 5. Synthetic Manifests ────────────────────────────────────────────────────

def _make_requirements(pkg: str, ver: str) -> tuple[str, str]:
    manifest = f"{pkg}=={ver}\n"
    safe_name = pkg.replace("-", "_").split("/")[0]
    snippet = textwrap.dedent(f"""\
        import {safe_name}

        client = {safe_name}.Client()
        result = client.process(data)
    """)
    return manifest, snippet


def _make_package_json(pkg: str, ver: str) -> tuple[str, str]:
    manifest = json.dumps({"name": "app", "version": "1.0.0", "dependencies": {pkg: ver}}, indent=2) + "\n"
    safe_name = pkg.lstrip("@").replace("-", "_").replace("/", "_").replace(".", "_").split("_")[0] or "lib"
    snippet = textwrap.dedent(f"""\
        const {safe_name} = require('{pkg}');

        const result = {safe_name}.execute(input);
    """)
    return manifest, snippet


def _make_pom(pkg: str, ver: str) -> tuple[str, str]:
    group, artifact = (pkg.split(":", 1) + [pkg])[:2]
    manifest = textwrap.dedent(f"""\
        <project>
          <dependencies>
            <dependency>
              <groupId>{group}</groupId>
              <artifactId>{artifact}</artifactId>
              <version>{ver}</version>
            </dependency>
          </dependencies>
        </project>
    """)
    snippet = textwrap.dedent(f"""\
        import {group.replace('-', '.')};

        public class App {{
            public static void main(String[] args) {{
                Client client = new Client();
                client.execute(data);
            }}
        }}
    """)
    return manifest, snippet


def _make_go_mod(pkg: str, ver: str) -> tuple[str, str]:
    if not ver.startswith("v"):
        ver = "v" + ver
    manifest = textwrap.dedent(f"""\
        module example.com/app

        go 1.21

        require (
            {pkg} {ver}
        )
    """)
    snippet = textwrap.dedent(f"""\
        package main

        import "{pkg}"

        func main() {{
            client := lib.NewClient()
            client.Do(ctx, request)
        }}
    """)
    return manifest, snippet


_RENDERERS = {
    "requirements.txt": ("PyPI", _make_requirements),
    "package.json": ("npm", _make_package_json),
    "pom.xml": ("Maven", _make_pom),
    "go.mod": ("Go", _make_go_mod),
}

_FALLBACKS: dict[str, list[tuple[str, str]]] = {
    "PyPI": [("requests", "2.25.0"), ("urllib3", "1.26.0"), ("cryptography", "3.3.0"), ("pillow", "8.1.0")],
    "npm": [("axios", "0.21.0"), ("lodash", "4.17.20"), ("express", "4.17.0"), ("marked", "2.0.0")],
    "Maven": [("org.apache.logging.log4j:log4j-core", "2.14.1"), ("com.fasterxml.jackson.core:jackson-databind", "2.12.0")],
    "Go": [("golang.org/x/net", "0.0.0-20210119194325"), ("github.com/gin-gonic/gin", "1.6.3")],
}


def _load_osv_pool() -> dict[str, list[dict]]:
    osv_path = OUTPUT_DIR / "osv.jsonl"
    pool: dict[str, list[dict]] = {eco: [] for eco in _FALLBACKS}
    if not osv_path.exists():
        return pool
    with osv_path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
                eco = rec.get("ecosystem", "")
                if eco not in pool:
                    continue
                for aff in rec.get("affected", []):
                    pkg = aff.get("package", "")
                    if not pkg:
                        continue
                    for rng in aff.get("ranges", []):
                        for evt in rng.get("events", []):
                            if "introduced" in evt and evt["introduced"] not in ("0", ""):
                                pool[eco].append({"package": pkg, "version": evt["introduced"], "osv_id": rec["id"]})
            except (json.JSONDecodeError, KeyError):
                pass
    return pool


def ingest_manifests():
    from tqdm import tqdm  # noqa: PLC0415
    out = OUTPUT_DIR / "manifests.jsonl"
    print(f"[Manifests] → {out}")
    pool = _load_osv_pool()
    random.seed(42)
    TARGET = 500
    records = []

    for manifest_type, (ecosystem, renderer) in _RENDERERS.items():
        candidates = pool.get(ecosystem, [])
        random.shuffle(candidates)
        sample = candidates[:TARGET]

        for entry in tqdm(sample, desc=f"Manifests/{manifest_type}"):
            try:
                manifest_content, snippet = renderer(entry["package"], entry["version"])
                records.append({
                    "source": "manifest",
                    "manifest_type": manifest_type,
                    "ecosystem": ecosystem,
                    "package": entry["package"],
                    "version": entry["version"],
                    "osv_id": entry["osv_id"],
                    "manifest_content": manifest_content,
                    "code_snippet": snippet,
                })
            except Exception:
                pass

        # Fill remaining slots from static fallbacks
        written = sum(1 for r in records if r.get("manifest_type") == manifest_type)
        fallbacks = _FALLBACKS.get(ecosystem, [])
        for pkg, ver in (fallbacks * ((TARGET // max(len(fallbacks), 1)) + 1))[: TARGET - written]:
            try:
                manifest_content, snippet = renderer(pkg, ver)
                records.append({
                    "source": "manifest_synthetic",
                    "manifest_type": manifest_type,
                    "ecosystem": ecosystem,
                    "package": pkg,
                    "version": ver,
                    "osv_id": "",
                    "manifest_content": manifest_content,
                    "code_snippet": snippet,
                })
            except Exception:
                pass

    n = _write_jsonl(out, records)
    print(f"[Manifests] {n} records written")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ingest_nvd()
    ingest_osv()
    ingest_ghsa()
    ingest_cvefixes()
    ingest_manifests()
    print(f"\nDone. Corpus files in {OUTPUT_DIR}/")
