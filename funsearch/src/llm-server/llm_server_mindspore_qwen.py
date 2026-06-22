"""Qwen MindFormers /completions server for FunSearch.

This server uses the vendored ``research/qwen`` MindFormers implementation
instead of the generic AutoModel path, because MindFormers r1.3.0 in ModelArts
does not ship the Qwen research YAML files.
"""

from __future__ import annotations

import gc
import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, jsonify, request
from flask_cors import CORS
from mindformers import MindFormerConfig, Trainer
from mindformers.core.context import build_context


REPO_ROOT = Path(__file__).resolve().parents[1]
QWEN_DIR = REPO_ROOT / "research" / "qwen"
if str(QWEN_DIR) not in sys.path:
    sys.path.insert(0, str(QWEN_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Register Qwen model/tokenizer/config classes with MindFormers.
import qwen_config  # noqa: F401,E402
import qwen_model  # noqa: F401,E402
import qwen_tokenizer  # noqa: F401,E402


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config", default=str(QWEN_DIR / "predict_qwen_7b_smoke.yaml"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11011)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--seq-length", type=int, default=64)
    parser.add_argument("--predict-length", type=int, default=8)
    parser.add_argument("--max-repeat-prompt", type=int, default=1)
    parser.add_argument("--warmup", action="store_true")
    return parser.parse_args()


def _normalise(result: Any, prompt: str) -> str:
    if isinstance(result, str):
        return result[len(prompt):] if result.startswith(prompt) else result
    if isinstance(result, list) and result:
        return _normalise(result[0], prompt)
    if isinstance(result, tuple) and result:
        return _normalise(result[0], prompt)
    if isinstance(result, dict):
        for key in ("text_generation_text", "generated_text", "text", "output"):
            if key in result:
                return _normalise(result[key], prompt)
    return str(result)


def _apply_generation_params(config: MindFormerConfig, params: Dict[str, Any], args) -> int:
    model_config = config.model.model_config
    model_config.seq_length = int(params.get("seq_length", args.seq_length))
    model_config.batch_size = 1
    model_config.use_past = bool(params.get("use_past", True))
    model_config.do_sample = bool(params.get("do_sample", False))
    model_config.top_k = int(params.get("top_k", 0) or 0)
    model_config.top_p = float(params.get("top_p", 0.8) or 0.8)
    model_config.max_decode_length = int(params.get("max_length", params.get("max_new_tokens", args.predict_length)))
    return int(params.get("max_length", params.get("max_new_tokens", args.predict_length)))


def create_app(args) -> Flask:
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    config = MindFormerConfig(str(config_path))
    config.context.device_id = args.device_id
    config.use_parallel = False
    config.model.model_config.seq_length = args.seq_length
    config.model.model_config.batch_size = 1
    config.model.model_config.use_past = True
    config.model.model_config.do_sample = False
    config.model.model_config.max_decode_length = args.predict_length

    build_context(config)
    task = Trainer(args=config, task="text_generation")

    app = Flask(__name__)
    CORS(app)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"ok": True, "backend": "mindformers-qwen", "config": str(config_path)})

    @app.route("/completions", methods=["POST"])
    def completions():
        content = request.get_json(force=True)
        prompt = str(content["prompt"]).strip()
        repeat_prompt = min(int(content.get("repeat_prompt", 1)), args.max_repeat_prompt)
        params = content.get("params", {}) or {}
        max_length = _apply_generation_params(config, params, args)

        responses: List[str] = []
        for _ in range(repeat_prompt):
            result = task.predict(input_data=[prompt], predict_checkpoint=None, max_length=max_length)
            responses.append(_normalise(result, prompt))

        gc.collect()
        return jsonify({"content": responses})

    if args.warmup:
        task.predict(input_data=["hello"], predict_checkpoint=None, max_length=args.predict_length)

    return app


if __name__ == "__main__":
    parsed = parse_args()
    create_app(parsed).run(host=parsed.host, port=parsed.port)
