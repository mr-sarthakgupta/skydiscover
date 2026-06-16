#!/usr/bin/env bash
set -euo pipefail

# Configuration (override via environment)
MAX_COST="${MAX_COST:-35}"
ITERATIONS="${ITERATIONS:-30}"
SEARCH="${SEARCH:-adaevolve}"  # supported: adaevolve, evox
ALLOW_SMOKE_FAIL="${ALLOW_SMOKE_FAIL:-0}"
ALLOW_RUNTIME_ASSET_FAIL="${ALLOW_RUNTIME_ASSET_FAIL:-0}"

# Keep Bedrock calls pinned to the repo-standard region.
AWS_REGION="us-east-1"
export AWS_REGION
export AWS_DEFAULT_REGION="$AWS_REGION"
export BEDROCK_AWS_REGION="$AWS_REGION"
export BEDROCK_CONNECT_TIMEOUT="${BEDROCK_CONNECT_TIMEOUT:-30}"
export BEDROCK_READ_TIMEOUT="${BEDROCK_READ_TIMEOUT:-1800}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCHMARK_DIR="benchmarks/protein_binder_design"
BENCHMARK_ROOT="$SCRIPT_DIR/$BENCHMARK_DIR"
PROTEINA_COMPLEXA_ROOT="${PROTEINA_COMPLEXA_ROOT:-$REPO_ROOT/Proteina-Complexa}"

# Load workspace-level overrides first, then Proteina runtime configuration.
if [ -f "$REPO_ROOT/.env" ]; then
    # shellcheck disable=SC1090
    set -a
    source "$REPO_ROOT/.env"
    set +a
fi
if [ -f "$SCRIPT_DIR/.env" ]; then
    # shellcheck disable=SC1090
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

# Proteina env.sh sources .env and maps UV tool paths.
if [ -f "$PROTEINA_COMPLEXA_ROOT/env.sh" ]; then
    # shellcheck disable=SC1090
    set -a
    source "$PROTEINA_COMPLEXA_ROOT/env.sh" >/dev/null
    set +a
elif [ -f "$PROTEINA_COMPLEXA_ROOT/.env" ]; then
    # shellcheck disable=SC1090
    set -a
    source "$PROTEINA_COMPLEXA_ROOT/.env"
    set +a
fi

# Canonical paths for this workspace layout.
export PROTEINA_COMPLEXA_ROOT
export LOCAL_CODE_PATH="${LOCAL_CODE_PATH:-$PROTEINA_COMPLEXA_ROOT}"
export CKPT_PATH="${CKPT_PATH:-$PROTEINA_COMPLEXA_ROOT/ckpts}"
export COMMUNITY_MODELS_PATH="${COMMUNITY_MODELS_PATH:-$PROTEINA_COMPLEXA_ROOT/community_models}"
export AF2_DIR="${AF2_DIR:-$COMMUNITY_MODELS_PATH/ckpts/AF2}"
export ESM_DIR="${ESM_DIR:-$COMMUNITY_MODELS_PATH/ckpts/ESM2}"
export RF3_DIR="${RF3_DIR:-$COMMUNITY_MODELS_PATH/ckpts/RF3}"
export RF3_CKPT_PATH="${RF3_CKPT_PATH:-$RF3_DIR/rf3_foundry_01_24_latest_remapped.ckpt}"
export RF3_EXEC_PATH="${RF3_EXEC_PATH:-$PROTEINA_COMPLEXA_ROOT/.venv/bin/rf3}"
export DATA_PATH="${DATA_PATH:-$PROTEINA_COMPLEXA_ROOT/data}"
export PROTEINA_PYTHON="${PROTEINA_PYTHON:-$PROTEINA_COMPLEXA_ROOT/.venv/bin/python}"
export PYTHONPATH="${PROTEINA_COMPLEXA_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
export SKYDISCOVER_BINDER_TIMEOUT="${SKYDISCOVER_BINDER_TIMEOUT:-9000}"
# 24 GB GPU defaults: keep stage-2 search small and force BestOfN chunking.
export SKYDISCOVER_BINDER_STAGE2_BATCH_SIZE="${SKYDISCOVER_BINDER_STAGE2_BATCH_SIZE:-2}"
export SKYDISCOVER_BINDER_STAGE2_NUM_LENGTH_SAMPLES="${SKYDISCOVER_BINDER_STAGE2_NUM_LENGTH_SAMPLES:-2}"
export SKYDISCOVER_BINDER_STAGE2_REPLICAS="${SKYDISCOVER_BINDER_STAGE2_REPLICAS:-2}"
export SKYDISCOVER_BINDER_MAX_BATCH_SIZE="${SKYDISCOVER_BINDER_MAX_BATCH_SIZE:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "============================================================"
echo "  Protein Binder Design - 3DI3 IL-7Ralpha"
echo "  SkyDiscover + Proteina-Complexa"
echo "============================================================"
echo ""
echo "  Search algorithm : $SEARCH"
echo "  Max iterations   : $ITERATIONS"
echo "  Cost budget      : \$$MAX_COST"
echo "  AWS region       : $AWS_REGION"
echo "  Benchmark root   : $BENCHMARK_ROOT"
echo "  Proteina root    : $PROTEINA_COMPLEXA_ROOT"
echo "  Checkpoint path  : $CKPT_PATH"
echo "  AF2 params       : $AF2_DIR"
echo "  Proteina Python  : $PROTEINA_PYTHON"
echo "  GPU batch caps   : stage2 batch=$SKYDISCOVER_BINDER_STAGE2_BATCH_SIZE length=$SKYDISCOVER_BINDER_STAGE2_NUM_LENGTH_SAMPLES replicas=$SKYDISCOVER_BINDER_STAGE2_REPLICAS max_batch=$SKYDISCOVER_BINDER_MAX_BATCH_SIZE"
echo ""

# Check Bedrock credentials.
if [ -z "${AWS_BEARER_TOKEN_BEDROCK:-}" ] && [ -z "${AWS_ACCESS_KEY_ID:-}" ] && [ -z "${AWS_PROFILE:-}" ] && [ ! -f "${AWS_SHARED_CREDENTIALS_FILE:-$HOME/.aws/credentials}" ]; then
    echo "ERROR: Bedrock credentials not found."
    echo "  Set AWS_BEARER_TOKEN_BEDROCK, AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY,"
    echo "  AWS_PROFILE, or provide ~/.aws/credentials."
    exit 1
fi
echo "[OK] Bedrock credentials detected"

cd "$SCRIPT_DIR"

echo ""
echo "Installing SkyDiscover dependencies..."
uv sync --extra bedrock --quiet 2>/dev/null || uv sync --extra bedrock
echo "[OK] SkyDiscover dependencies installed"

echo ""
echo "Checking benchmark assets..."
uv run python - <<'PY'
from pathlib import Path
import json

base = Path("benchmarks/protein_binder_design/assets")
required = [
    "3di3.pdb",
    "3di3.cif",
    "3di3_chain_b_il7ra_target.pdb",
    "target_metadata.json",
]
missing = [name for name in required if not (base / name).exists()]
if missing:
    raise SystemExit(f"missing benchmark assets: {', '.join(missing)}")

meta = json.loads((base / "target_metadata.json").read_text())
target_path = base / meta["target"]["target_pdb"]
seen = []
residue_keys = set()
chains = set()
for line in target_path.read_text().splitlines():
    if not line.startswith("ATOM  "):
        continue
    chain = line[21].strip()
    resid = int(line[22:26])
    icode = line[26].strip()
    chains.add(chain)
    key = (chain, resid, icode)
    if key not in residue_keys:
        residue_keys.add(key)
        seen.append(f"{chain}{resid}{icode}")

if chains != {"B"}:
    raise SystemExit(f"target PDB must contain only chain B ATOM records; found {sorted(chains)}")
if seen != meta["target"]["allowed_hotspots"]:
    raise SystemExit("target PDB residues do not match metadata allowed_hotspots")
if meta["target"].get("proteina_target_input") != "B17-209":
    raise SystemExit("metadata proteina_target_input should be B17-209")
if not all(h in seen for h in meta["prior_art_hotspots"]["hotspots"]):
    raise SystemExit("one or more prior hotspots are absent from the target")

print(f"[OK] 3DI3 chain B target: {len(seen)} residues, {seen[0]}-{seen[-1]}")
PY

echo ""
echo "Checking Proteina-Complexa runtime assets..."
runtime_missing=0
if [ ! -d "$PROTEINA_COMPLEXA_ROOT" ]; then
    echo "ERROR: Proteina-Complexa repo not found: $PROTEINA_COMPLEXA_ROOT"
    runtime_missing=1
elif [ ! -f "$PROTEINA_COMPLEXA_ROOT/configs/search_binder_local_pipeline.yaml" ]; then
    echo "ERROR: Proteina binder pipeline config not found under $PROTEINA_COMPLEXA_ROOT"
    runtime_missing=1
fi

if [ ! -x "$PROTEINA_PYTHON" ]; then
    echo "ERROR: Proteina Python not found or not executable: $PROTEINA_PYTHON"
    runtime_missing=1
fi

if [ ! -f "$CKPT_PATH/complexa.ckpt" ]; then
    echo "ERROR: missing checkpoint: $CKPT_PATH/complexa.ckpt"
    runtime_missing=1
fi
if [ ! -f "$CKPT_PATH/complexa_ae.ckpt" ]; then
    echo "ERROR: missing checkpoint: $CKPT_PATH/complexa_ae.ckpt"
    runtime_missing=1
fi
if [ -z "${AF2_DIR:-}" ] || [ ! -d "${AF2_DIR:-}" ]; then
    echo "ERROR: AF2_DIR is not set to an existing AlphaFold params directory"
    runtime_missing=1
elif [ ! -f "$AF2_DIR/params_model_5_ptm.npz" ]; then
    echo "ERROR: AF2 params look incomplete under $AF2_DIR"
    runtime_missing=1
fi

if [ "$runtime_missing" = "1" ]; then
    if [ "$ALLOW_RUNTIME_ASSET_FAIL" = "1" ]; then
        echo "WARNING: runtime asset check failed - continuing because ALLOW_RUNTIME_ASSET_FAIL=1"
    else
        echo "Set CKPT_PATH and AF2_DIR, or run:"
        echo "  cd $PROTEINA_COMPLEXA_ROOT && .venv/bin/complexa download --everything"
        echo "Or set ALLOW_RUNTIME_ASSET_FAIL=1 to bypass this preflight."
        exit 1
    fi
else
    echo "[OK] Proteina runtime assets detected"
fi

echo ""
echo "Smoke test: validating seed binder-design run spec..."
if SKYDISCOVER_BINDER_VALIDATE_ONLY=1 uv run python "$BENCHMARK_DIR/evaluator.py" "$BENCHMARK_DIR/initial_program.py"; then
    echo "[OK] Smoke test passed"
elif [ "$ALLOW_SMOKE_FAIL" = "1" ]; then
    echo "WARNING: Smoke test failed - continuing anyway (ALLOW_SMOKE_FAIL=1)"
else
    echo "ERROR: Smoke test failed. Fix evaluator/assets before launching a costed LLM run."
    echo "  Set ALLOW_SMOKE_FAIL=1 to override."
    exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-$(uv run python -c "
from skydiscover.config import build_output_dir
import os
print(build_output_dir('${SEARCH}', os.path.abspath('${BENCHMARK_DIR}/initial_program.py'), base_dir='outputs_binder_design'))
")}"
mkdir -p "$OUTPUT_DIR/reference"

# Agentic path validation resolves symlinks, so use the real benchmark tree.
export BENCHMARK_CODEBASE_ROOT="$BENCHMARK_ROOT"

echo ""
echo "============================================================"
echo "  Starting discovery run"
echo "  Target           : RCSB 3DI3 chain B, human IL-7Ralpha"
echo "  Output directory : $OUTPUT_DIR"
echo "  Reference files  : $OUTPUT_DIR/reference"
echo "============================================================"
echo ""

uv run skydiscover-run \
    "$BENCHMARK_DIR/initial_program.py" \
    "$BENCHMARK_DIR/evaluator.py" \
    -c "$BENCHMARK_DIR/config.yaml" \
    -s "$SEARCH" \
    -i "$ITERATIONS" \
    --max-cost "$MAX_COST" \
    -o "$OUTPUT_DIR"

echo ""
echo "Done."
