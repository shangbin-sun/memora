#!/usr/bin/env bash
# 一键评测 Memora：临时 DB → 启动服务 → 写入/召回 → LLM 打分。
# Usage (WSL):
#   ./scripts/run_eval.sh
#   ./scripts/run_eval.sh --limit 30
#   ./scripts/run_eval.sh --datasets cockpit-pref-recall.json cockpit-conflict-update.json
#   ./scripts/run_eval.sh --no-llm
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1090
source "$HOME/.cargo/env" 2>/dev/null || true
# Load .env without breaking on CRLF (Windows-checked-out files).
# Strip inline comments (# ...) and surrounding whitespace from values.
if [[ -f "$ROOT/.env" ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
    if [[ "$line" == *=* ]]; then
      key="${line%%=*}"
      val="${line#*=}"
      key="${key%"${key##*[![:space:]]}"}"
      key="${key#"${key%%[![:space:]]*}"}"
      if [[ "$val" != \"*\" && "$val" != \'*\' ]]; then
        val="${val%%#*}"
      fi
      val="${val%"${val##*[![:space:]]}"}"
      val="${val#"${val%%[![:space:]]*}"}"
      if [[ "$val" == \"*\" && "$val" == *\" ]]; then
        val="${val:1:${#val}-2}"
      elif [[ "$val" == \'*\' && "$val" == *\' ]]; then
        val="${val:1:${#val}-2}"
      fi
      export "$key=$val"
    fi
  done < "$ROOT/.env"
fi

BIND="${MEMORA_EVAL_BIND:-127.0.0.1:8790}"
DB_REL="${MEMORA_EVAL_DB:-data/memora_test.db}"
DB_PATH="$ROOT/$DB_REL"
OUT_DIR="${MEMORA_EVAL_OUT:-$ROOT/tests/eval_out}"
NAMESPACE="${MEMORA_EVAL_NAMESPACE:-eval}"
BASE_URL="http://${BIND}"
PID_FILE="$OUT_DIR/server.pid"
LOG_FILE="$OUT_DIR/server.log"

mkdir -p "$(dirname "$DB_PATH")" "$OUT_DIR"

# --- clean temp db ---
echo "==> reset db: $DB_PATH"
rm -f "$DB_PATH" "${DB_PATH}-wal" "${DB_PATH}-shm"

# --- resolve server binary ---
resolve_server() {
  local need_build=0
  if [[ ! -x "$ROOT/target/debug/memora-server" && ! -x "$ROOT/target/release/memora-server" ]]; then
    need_build=1
  fi
  # Prefer release if present; otherwise ensure debug is fresh enough vs source rename LLM_*.
  if [[ -x "$ROOT/target/release/memora-server" ]]; then
    echo "$ROOT/target/release/memora-server"
    return
  fi
  if [[ "$need_build" -eq 1 ]] || [[ "$ROOT/src/qianfan.rs" -nt "$ROOT/target/debug/memora-server" ]]; then
    echo "building memora-server (debug)..." >&2
    cargo build -p memora-server >&2
  fi
  echo "$ROOT/target/debug/memora-server"
}

SERVER_BIN="$(resolve_server)"

cleanup() {
  if [[ -f "$PID_FILE" ]]; then
    local pid
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      echo "==> stop server pid=$pid"
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
  fi
}
trap cleanup EXIT INT TERM

# --- start server ---
echo "==> start server: $SERVER_BIN"
echo "    bind=$BIND db=$DB_REL namespace=$NAMESPACE"
MEMORA_BIND="$BIND" \
MEMORA_DB="$DB_PATH" \
MEMORA_NAMESPACE="$NAMESPACE" \
  "$SERVER_BIN" >"$LOG_FILE" 2>&1 &
echo $! >"$PID_FILE"
echo "    pid=$(cat "$PID_FILE") log=$LOG_FILE"

# --- run eval ---
echo "==> run eval"
python3 "$ROOT/scripts/eval_memory.py" \
  --base-url "$BASE_URL" \
  --out-dir "$OUT_DIR" \
  --namespace "$NAMESPACE" \
  --wait-health \
  "$@"

echo "==> artifacts"
echo "    $OUT_DIR/result.json"
echo "    $OUT_DIR/report.json"
echo "    $OUT_DIR/report.md"
