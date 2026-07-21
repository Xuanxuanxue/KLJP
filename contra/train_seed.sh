#!/bin/bash
# The script works from any checkout; paths are resolved by setting.py.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${KLJP_MODE:-auto}"

for seed in 5 17 67 76 81; do
    python "${SCRIPT_DIR}/train_bl.py" --mode "${MODE}" --seed "${seed}"
done
