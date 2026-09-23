"""Offline JSONL bridge for a warm laya-coreml 0.1.0 model.

Protocol is deliberately narrow: one choice among host supplied opaque labels.
The Rust host still owns candidate eligibility and execution.
"""

import json
import os
import sys
import urllib.parse
import urllib.request
import warnings


MODEL_REPO = "aac6fef/laya-multilingual-coreml"
MODEL_REVISION = "8139e9089273319512c730218903784074133187"
MODEL_FILES = [
    ("model.mlpackage/Manifest.json", 617),
    ("model.mlpackage/Data/com.apple.CoreML/weights/weight.bin", 643966208),
    ("model.mlpackage/Data/com.apple.CoreML/model.mlmodel", 1536512),
    ("encoder/config.json", 1938),
    ("tokenizer/tokenizer_config.json", 524),
    ("tokenizer/tokenizer.json", 34363188),
    ("coreml_config.json", 2121),
    ("rl_agent_config.json", 472),
]
MODEL_BYTES = sum(size for _, size in MODEL_FILES)


def emit(message):
    # Core ML can write native diagnostics to fd 1 without a newline.
    sys.stdout.write("\nKEEL_LAYA_JSON:" + json.dumps(message, separators=(",", ":"), allow_nan=False) + "\n")
    sys.stdout.flush()


def fits_without_truncation(agent, state, question):
    """Laya normally clips the question prefix/state; refuse that for routing."""
    from laya_coreml.common import render_options

    q = agent._to_internal(question)
    tok = agent.tok
    options = render_options(q)
    if len(options) > agent.shape["max_options"]:
        return False
    clean = lambda value: value.replace(tok.mask_token, " ")
    head = tok("choice question: " + clean(q["ins"]), add_special_tokens=False)["input_ids"]
    option_lengths = [
        len(tok(" " + clean(option), add_special_tokens=False)["input_ids"])
        for option in options
    ]
    if any(length > 48 for length in option_lengths):
        return False
    # CLS, the two SEP markers, and one MASK marker per option.
    prefix_length = 3 + len(head) + len(options) + sum(option_lengths)
    if 2 + len(head) + len(options) + sum(option_lengths) > agent.cfg.get("head_max_len", 192):
        return False
    state_length = len(tok(clean(state), add_special_tokens=False)["input_ids"])
    return prefix_length + state_length <= min(
        agent.cfg.get("max_len", 512), agent.shape["max_length"]
    )


def download_model(output):
    downloaded = 0
    emit({"downloaded_bytes": downloaded, "total_bytes": MODEL_BYTES})
    for relative, expected_size in MODEL_FILES:
        destination = os.path.join(output, *relative.split("/"))
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        path = urllib.parse.quote(relative, safe="/")
        url = f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{path}"
        request = urllib.request.Request(url, headers={"User-Agent": "Keel-Laya/0.1"})
        file_bytes = 0
        with urllib.request.urlopen(request, timeout=60) as response, open(destination, "wb") as target:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
                file_bytes += len(chunk)
                downloaded += len(chunk)
                emit({"downloaded_bytes": downloaded, "total_bytes": MODEL_BYTES})
        if file_bytes != expected_size:
            raise ValueError("download_size_mismatch")
    if downloaded != MODEL_BYTES:
        raise ValueError("download_size_mismatch")


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "--capabilities":
        print("download-model-v1 download-progress-v1")
        return 0
    if len(sys.argv) == 4 and sys.argv[1:3] == ["--download-model", "--output"]:
        try:
            download_model(sys.argv[3])
            return 0
        except Exception:
            return 2
    if len(sys.argv) == 3 and sys.argv[1] == "--model":
        model_dir = sys.argv[2]
    elif len(sys.argv) == 2:
        model_dir = sys.argv[1]
    else:
        emit({"ready": False, "error": "model_unavailable"})
        return 2
    try:
        import laya_coreml as laya

        agent = laya.load(model_dir, local_files_only=True)
    except Exception:
        emit({"ready": False, "error": "model_unavailable"})
        return 2
    emit({"ready": True, "model": "laya-coreml"})
    for line in sys.stdin:
        try:
            request = json.loads(line)
            state = request["state"]
            pairs = request["criteria"]
            if not isinstance(state, str) or not isinstance(pairs, list):
                raise ValueError("invalid_request")
            criteria = dict(pairs)
            if len(criteria) != len(pairs) or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in criteria.items()
            ):
                raise ValueError("invalid_request")
            question = {
                "type": "choice",
                "instructions": "Choose the prepared step that best serves the task. Pick escalate if none fits.",
                "criteria": criteria,
            }
            if not fits_without_truncation(agent, state, question):
                emit({"error": "capacity"})
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                answer = agent.predict(state, {"select": question})["answers"]["select"]
            emit({
                "choice": answer["choice"],
                "probabilities": answer["probabilities"],
                "confidence": answer["confidence"],
            })
        except Exception:
            # Never send exception text: it may include the task or local paths.
            emit({"error": "inference"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
