#!/usr/bin/env bash
DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$DIR/.venv/bin/python" "$DIR/cleanframe.py" "$@"
