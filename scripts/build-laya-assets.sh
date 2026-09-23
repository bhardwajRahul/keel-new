#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python="${PYTHON_FOR_LAYA:-python3.12}"
build_root="${1:-$root/build/laya}"
mkdir -p "$build_root"
"$python" -m venv "$build_root/venv"
"$build_root/venv/bin/python" -m pip install --disable-pip-version-check 'laya-coreml==0.1.0' 'pyinstaller==6.22.3'
"$build_root/venv/bin/hf" download aac6fef/laya-multilingual-coreml \
  --revision 8139e9089273319512c730218903784074133187 \
  --local-dir "$build_root/model"
"$build_root/venv/bin/pyinstaller" --noconfirm --onedir --name laya-worker \
  --collect-all laya_coreml --collect-all coremltools \
  --collect-all tokenizers --collect-all safetensors \
  --distpath "$build_root/dist" --workpath "$build_root/pyi-build" \
  --specpath "$build_root/pyi-spec" "$root/crates/laya-local/worker.py"
echo "KEEL_LAYA_WORKER_BUNDLE=$build_root/dist/laya-worker"
echo "KEEL_LAYA_MODEL=$build_root/model"
