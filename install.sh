#!/usr/bin/env bash
# Clean Frame installer (Linux)
set -e
cd "$(dirname "$0")"
MIRROR=${PIP_MIRROR:-https://mirrors.aliyun.com/pypi/simple/}
python3 -m venv .venv
.venv/bin/python -m pip install -U pip -i "$MIRROR"
.venv/bin/python -m pip install -r requirements.txt -i "$MIRROR"
.venv/bin/python -c "import onnxruntime as o; o.preload_dlls(); print(o.get_available_providers())"
echo "Install OK"
