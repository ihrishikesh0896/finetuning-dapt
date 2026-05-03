#!/usr/bin/env bash
# run_pipeline.sh — End-to-end VulnReach DAPT → SFT pipeline with timing logs.
#
# Usage:
#   bash run_pipeline.sh                         # run all phases
#   bash run_pipeline.sh --install-deps          # pip install first, then run
#   bash run_pipeline.sh --skip-ingest           # skip corpus download (already done)
#   bash run_pipeline.sh --from-phase 3          # start from a specific phase number
#
# Phases:
#   1   ingest_corpus.py       — download NVD / OSV / GHSA / CVEfixes
#   1b  merge_and_tokenize.py  — merge JSONL + tokenize to HF arrow
#   3   build_sft_dataset.py   — build instruction-tuning dataset
#   2   train_dapt.py          — DAPT continued pre-training (CUDA required)
#   4   train_sft.py           — SFT instruction fine-tuning  (CUDA required)
#   5   quantize.py            — merge adapters + export GGUF
#   5t  test_inference.py      — smoke-test the GGUF on 3 eval samples

set -uo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

# ── Defaults ──────────────────────────────────────────────────────────────────
INSTALL_DEPS=0
SKIP_INGEST=0
FROM_PHASE=0          # 0 = run all
LOG_DIR="logs"
PYTHON="${PYTHON:-python3}"

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --install-deps)  INSTALL_DEPS=1 ;;
    --skip-ingest)   SKIP_INGEST=1 ;;
    --from-phase)    FROM_PHASE="$2"; shift ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
  shift
done

# ── Helpers ───────────────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"

# Timing state
declare -A PHASE_SECS
declare -A PHASE_STATUS

_ts()     { date +%s; }
_hms()    {                          # seconds → H:MM:SS
  local s=$1
  printf "%d:%02d:%02d" $((s/3600)) $((s%3600/60)) $((s%60))
}
_now()    { date '+%Y-%m-%d %H:%M:%S'; }
_log()    { echo -e "${CYAN}[$(_now)]${RESET} $*"; }
_ok()     { echo -e "${GREEN}[OK]${RESET} $*"; }
_warn()   { echo -e "${YELLOW}[WARN]${RESET} $*"; }
_err()    { echo -e "${RED}[FAIL]${RESET} $*" >&2; }
_banner() {
  echo ""
  echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
  echo -e "${BOLD}${CYAN}  Phase $1 — $2${RESET}"
  echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
}

_run_phase() {
  local phase_id="$1"   # e.g. "1", "1b", "2" ...
  local label="$2"
  local script="$3"
  local logfile="$LOG_DIR/phase_${phase_id}.log"

  _banner "$phase_id" "$label"

  if [[ ! -f "$script" ]]; then
    _warn "$script not found — skipping phase $phase_id"
    PHASE_STATUS[$phase_id]="SKIP"
    PHASE_SECS[$phase_id]=0
    return 0
  fi

  _log "Script : $script"
  _log "Log    : $logfile"
  echo ""

  local t0; t0=$(_ts)

  # Run with live output AND save to log file
  if $PYTHON "$script" 2>&1 | tee "$logfile"; then
    local elapsed=$(( $(_ts) - t0 ))
    PHASE_STATUS[$phase_id]="OK"
    PHASE_SECS[$phase_id]=$elapsed
    _ok "Phase $phase_id done in $(_hms $elapsed)"
  else
    local elapsed=$(( $(_ts) - t0 ))
    PHASE_STATUS[$phase_id]="FAIL"
    PHASE_SECS[$phase_id]=$elapsed
    _err "Phase $phase_id FAILED after $(_hms $elapsed) — see $logfile"
    echo ""
    echo -e "${RED}Last 20 lines of log:${RESET}"
    tail -20 "$logfile"
    echo ""
    _err "Pipeline stopped at phase $phase_id. Fix the error and re-run with --from-phase $phase_id"
    _print_summary
    exit 1
  fi
}

_phase_enabled() {
  local num="$1"   # numeric phase (1, 2, 3, 4, 5)
  [[ $FROM_PHASE -eq 0 ]] || [[ $num -ge $FROM_PHASE ]]
}

_print_summary() {
  echo ""
  echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
  echo -e "${BOLD}  Pipeline Summary${RESET}"
  echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
  printf "  %-6s %-32s %-8s %s\n" "Phase" "Description" "Status" "Duration"
  printf "  %-6s %-32s %-8s %s\n" "-----" "-----------" "------" "--------"

  local order=("1" "1b" "3" "2" "4" "5" "5t")
  local labels=(
    "Corpus ingestion"
    "Merge + tokenize"
    "SFT dataset build"
    "DAPT training"
    "SFT training"
    "Quantize → GGUF"
    "Inference smoke test"
  )

  local total=0
  for i in "${!order[@]}"; do
    local pid="${order[$i]}"
    local lbl="${labels[$i]}"
    local status="${PHASE_STATUS[$pid]:-SKIP}"
    local secs="${PHASE_SECS[$pid]:-0}"
    total=$((total + secs))

    local colour="$RESET"
    [[ "$status" == "OK"   ]] && colour="$GREEN"
    [[ "$status" == "FAIL" ]] && colour="$RED"
    [[ "$status" == "SKIP" ]] && colour="$YELLOW"

    printf "  %-6s %-32s ${colour}%-8s${RESET} %s\n" \
      "$pid" "$lbl" "$status" "$(_hms $secs)"
  done

  echo -e "  ${BOLD}Total wall time : $(_hms $total)${RESET}"
  echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
  echo ""
}

# ── Preflight ─────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}VulnReach DAPT Pipeline — $(_now)${RESET}"
echo -e "Python : $($PYTHON --version 2>&1)"
echo -e "CWD    : $(pwd)"
echo ""

# Check we're in repo root
if [[ ! -f "PROMPT.md" ]]; then
  _err "Run this script from the repo root (where PROMPT.md lives)."
  exit 1
fi

# Install deps
if [[ $INSTALL_DEPS -eq 1 ]]; then
  _log "Installing dependencies ..."
  $PYTHON -m pip install --quiet \
    requests tqdm gitpython datasets transformers sentencepiece protobuf \
    2>&1 | tee "$LOG_DIR/install.log"
  _ok "Dependencies installed"
fi

# GPU check — warn but don't abort (training scripts do their own hard check)
GPU_OK=0
if $PYTHON -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  GPU_NAME=$($PYTHON -c "import torch; print(torch.cuda.get_device_name(0))")
  GPU_MEM=$($PYTHON -c "import torch; print(round(torch.cuda.get_device_properties(0).total_memory/1e9,1))")
  _ok "GPU detected: $GPU_NAME (${GPU_MEM} GB)"
  GPU_OK=1
else
  _warn "No CUDA GPU detected — training phases will fail."
  _warn "Corpus ingestion and dataset build phases will still run."
fi

# ── Phase 1 — Corpus ingestion ────────────────────────────────────────────────
if [[ $SKIP_INGEST -eq 1 ]]; then
  _warn "Skipping Phase 1 (--skip-ingest)"
  PHASE_STATUS["1"]="SKIP"; PHASE_SECS["1"]=0
elif _phase_enabled 1; then
  _run_phase "1" "Corpus ingestion" "scripts/dapt/ingest_corpus.py"
else
  PHASE_STATUS["1"]="SKIP"; PHASE_SECS["1"]=0
fi

# ── Phase 1b — Merge + tokenize ───────────────────────────────────────────────
if _phase_enabled 1; then
  _run_phase "1b" "Merge + tokenize" "scripts/dapt/merge_and_tokenize.py"
else
  PHASE_STATUS["1b"]="SKIP"; PHASE_SECS["1b"]=0
fi

# ── Phase 3 — SFT dataset build ───────────────────────────────────────────────
if _phase_enabled 3; then
  _run_phase "3" "SFT dataset build" "scripts/sft/build_sft_dataset.py"
else
  PHASE_STATUS["3"]="SKIP"; PHASE_SECS["3"]=0
fi

# ── Phase 2 — DAPT training ───────────────────────────────────────────────────
if [[ $GPU_OK -eq 0 ]]; then
  _warn "Skipping Phase 2 (no CUDA GPU)"
  PHASE_STATUS["2"]="SKIP"; PHASE_SECS["2"]=0
elif _phase_enabled 2; then
  _run_phase "2" "DAPT training (10 000 steps)" "scripts/dapt/train_dapt.py"
else
  PHASE_STATUS["2"]="SKIP"; PHASE_SECS["2"]=0
fi

# ── Phase 4 — SFT training ────────────────────────────────────────────────────
if [[ $GPU_OK -eq 0 ]]; then
  _warn "Skipping Phase 4 (no CUDA GPU)"
  PHASE_STATUS["4"]="SKIP"; PHASE_SECS["4"]=0
elif _phase_enabled 4; then
  _run_phase "4" "SFT training (2 000 steps)" "scripts/sft/train_sft.py"
else
  PHASE_STATUS["4"]="SKIP"; PHASE_SECS["4"]=0
fi

# ── Phase 5 — Quantize ────────────────────────────────────────────────────────
if _phase_enabled 5; then
  _run_phase "5" "Quantize → GGUF" "scripts/export/quantize.py"
else
  PHASE_STATUS["5"]="SKIP"; PHASE_SECS["5"]=0
fi

# ── Phase 5t — Inference smoke test ──────────────────────────────────────────
if _phase_enabled 5; then
  _run_phase "5t" "Inference smoke test" "scripts/export/test_inference.py"
else
  PHASE_STATUS["5t"]="SKIP"; PHASE_SECS["5t"]=0
fi

# ── Final summary ─────────────────────────────────────────────────────────────
_print_summary
_ok "All done. Logs in $LOG_DIR/"
