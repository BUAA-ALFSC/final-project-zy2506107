#!/usr/bin/env bash
set -euo pipefail

REPO=/home/ma-user/work/codex_funsearch/rayzhhh-funsearch
RESULT=/home/ma-user/work/qwen_real_funsearch_formal_$(date +%Y%m%d_%H%M%S)

mkdir -p "$RESULT"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export QWEN_SAMPLE_LOG="$RESULT/qwen_raw_samples.jsonl"

curl -fsS http://127.0.0.1:11011/health > "$RESULT/qwen_health.json"

rm -rf logs/funsearch_qwen_real_formal
python /home/ma-user/work/run_qwen_funsearch_formal.py \
  > "$RESULT/funsearch_qwen_real_formal.log" 2>&1

cp -r logs/funsearch_qwen_real_formal "$RESULT/" 2>/dev/null || true
cp /home/ma-user/work/run_qwen_funsearch_formal.py "$RESULT/"
cp "$REPO/funsearch_bin_packing_local_llm.py" "$RESULT/"
tail -n 3000 "$(cat /home/ma-user/work/qwen_real_newenv_logs/latest_qwen_server_log.txt)" > "$RESULT/qwen_server_tail.log"

echo "$RESULT" > /home/ma-user/work/latest_qwen_funsearch_formal_result.txt
echo "RESULT_DIR:$RESULT"
