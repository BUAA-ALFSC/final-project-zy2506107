#!/usr/bin/env bash
set -euo pipefail

REPO=/home/ma-user/work/codex_funsearch/rayzhhh-funsearch
PORT=${PORT:-11011}
LOG_DIR=/home/ma-user/work/qwen_real_newenv_logs

mkdir -p "$LOG_DIR"
cd "$REPO"

pkill -f 'llm_server_mindspore_qwen.py' 2>/dev/null || true

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"

LOG_PATH="$LOG_DIR/qwen_server_$(date +%Y%m%d_%H%M%S).log"
nohup python llm-server/llm_server_mindspore_qwen.py \
  --config research/qwen/predict_qwen_7b_smoke.yaml \
  --host 127.0.0.1 \
  --port "$PORT" \
  --device-id 0 \
  --seq-length 128 \
  --predict-length 64 \
  > "$LOG_PATH" 2>&1 &

echo "$!" > "$LOG_DIR/qwen_server.pid"
echo "$LOG_PATH" > "$LOG_DIR/latest_qwen_server_log.txt"
echo "SERVER_PID:$!"
echo "SERVER_LOG:$LOG_PATH"
