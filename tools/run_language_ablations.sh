#!/usr/bin/env bash
# The 30 pilot commands (15 cultural, 15 universal) through every goal mode (D-038 / D-039): which English
# queries would reach the detector. Needs the Qolda server (not for mt_word); NLLB runs on the CPU.
#   bash tools/wsl.sh bash tools/run_language_ablations.sh
set -uo pipefail
cd "$(dirname "$0")/.."
for g in multi top1 mt_word mt_command; do
    echo "=== parse goal=${g}"
    python tools/eval_language.py parse "goal=${g}" 2>&1 | grep -E "^\| (all|cultural|universal) |^out:|Error|Traceback"
done
