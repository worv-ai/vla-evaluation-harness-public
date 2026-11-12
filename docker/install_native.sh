#!/usr/bin/env bash
set -euo pipefail
profile="${1:?native environment name required}"
pixi install --locked --manifest-path /opt/pixi/pixi.toml --environment "$profile" \
  --concurrent-downloads 4 --concurrent-solves 2
mkdir -p /usr/local/share/vla-build
cp /opt/pixi/pixi.toml /opt/pixi/pixi.lock /usr/local/share/vla-build/
printf '%s\n' "$profile" > /usr/local/share/vla-build/pixi-environment.txt
rm -rf /tmp/pixi-cache
