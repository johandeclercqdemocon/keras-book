#!/usr/bin/env bash
# Run every script in scripts/rows/ and write its output to scripts/rows/output/<name>.txt.
# Run from the book root, with the interpreter that has Keras: PYTHON=.venv/bin/python scripts/rows/run_all.sh
set -u
PY=${PYTHON:-python}
for f in scripts/rows/ch*.py; do
    name=$(basename "$f" .py)
    PYTHONPATH=. KERAS_BACKEND=tensorflow TF_CPP_MIN_LOG_LEVEL=3 "$PY" "$f" < /dev/null 2>/dev/null > "scripts/rows/output/$name.txt"
    echo "$name exit $?"
done
