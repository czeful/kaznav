#!/usr/bin/env bash
# Parse prompts on the DEV commands only (D-043): the 30 pilot commands and the 12 debug_ycb commands, prompt v1 vs v2,
# for one backend (its server running). Test commands (HM3D / OVON) are never used to choose a prompt.
#   bash tools/wsl.sh bash tools/run_prompt_dev.sh qolda        (or qwen3vl_4b)
set -uo pipefail
cd "$(dirname "$0")/.."
B="${1:-qolda}"
for v in v1 v2; do
    for cmds in "qolda check/data/commands_kk.json" commands/dev/debug_ycb_kk.json; do
        echo "=== ${B} parse_goal/${v} on ${cmds}"
        python tools/eval_language.py parse --commands "$cmds" "backend=${B}" "backend.prompts.parse_goal=default/parse_goal/${v}" 2>&1 \
            | grep -E "^\| (all|cultural|universal) |^out:|Error|Traceback"
    done
done
