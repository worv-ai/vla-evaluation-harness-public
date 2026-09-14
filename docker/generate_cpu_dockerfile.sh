#!/usr/bin/env bash
# Reuse each benchmark's runtime paths, native libraries and entrypoint.
set -euo pipefail
recipe="${1:?benchmark Dockerfile required}"
cat "$(dirname "${BASH_SOURCE[0]}")/cpu.Dockerfile"
sed -n '\|^RUN python /usr/local/lib/vla/export_runtime.py |s@export_runtime.py @export_runtime.py --inventory-only @p' "$recipe"
printf '\n'
awk '/^FROM \$\{RUNTIME_IMAGE\} AS runtime/ { runtime = 1 } runtime { print }' "$recipe" \
  | sed 's/^ARG IMAGE_PROFILE=gpu$/ARG IMAGE_PROFILE=cpu/' \
  | awk '
      /^COPY --from=builder \/runtime-root/ {
        path = $4
        stage = (path == "/app" || path == "/workspace" || path == "/opt/venv" || path ~ /^\/opt\/pixi\/\.pixi\/envs\// || path == "/usr/local/share/vla-build") ? "builder" : "gpu"
        print "COPY --from=" stage " " path " " path
        next
      }
      { print }
    '

