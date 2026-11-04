#!/usr/bin/env bash
# MAVIS - setup + start (macOS / Linux). Needs python3.11.
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  python3.11 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip >/dev/null
  .venv/bin/python -m pip install --no-deps -r requirements-lock.txt
fi
[ -f keys.env ] || cp keys.env.example keys.env
set -a; . ./keys.env; set +a
exec .venv/bin/python -m hub "$@"
