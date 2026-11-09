#!/usr/bin/env bash
# Install a pip lock file from env/ exactly: every line with --no-deps, a "+cuXXX" build from the PyTorch index of
# that CUDA version, everything else from PyPI (a PyPI torch and a "+cu" torch are different wheels).
# --force-reinstall: a line whose version conda already installed is still pip's wheel, as on the dev machine
# (imageio-ffmpeg: the conda package has no ffmpeg binary, the wheel bundles one; INSTALL_LOG.md, entry 6).
#   bash docker/pip_lock_install.sh /opt/qolda-env/bin/python env/qolda-env.pip.txt
set -euo pipefail
py="$1"; lock="$2"
tmp="$(mktemp -d)"
grep -v '^#' "$lock" | grep -v '+cu' > "$tmp/pypi.txt" || true
"$py" -m pip install --no-cache-dir --no-deps --force-reinstall -r "$tmp/pypi.txt"
for cu in $(grep -o '+cu[0-9]*' "$lock" | sort -u | tr -d '+'); do
    grep "+$cu\$" "$lock" > "$tmp/$cu.txt"
    "$py" -m pip install --no-cache-dir --no-deps --force-reinstall --index-url "https://download.pytorch.org/whl/$cu" \
        -r "$tmp/$cu.txt"
done
rm -rf "$tmp"
