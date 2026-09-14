#!/usr/bin/env bash
# Split a built CPU/GPU image into TAG-runtime and TAG-assets, without resolving dependencies.
# Usage: docker/package_runtime.sh BENCHMARK SOURCE_IMAGE OUTPUT_TAG
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
bench="${1:?benchmark required}"
source_image="${2:?source image required}"
output="${3:?output image tag required}"
source docker/images.sh
[[ " ${CPU_BENCHMARKS[*]} " == *" $bench "* ]] || { echo "Unsupported benchmark: $bench" >&2; exit 1; }
source_id=$(docker image inspect "$source_image" --format '{{.Id}}')
recipe=$(mktemp /tmp/vla-package-XXXXXX.Dockerfile)
paths=$(mktemp /tmp/vla-assets-XXXXXX.txt)
source_pin="vla-eval-packaging-source:${source_id#sha256:}-$$"
docker tag "$source_id" "$source_pin"
trap 'rm -f "$recipe" "$paths"; docker image rm "$source_pin" >/dev/null' EXIT
# Inspect an existing local artifact; never start its default simulator command.
docker run --rm --pull=never --runtime=runc --network none \
    --entrypoint python -v "$PWD/docker/split_assets.py:/tmp/split_assets.py:ro" \
    "$source_id" /tmp/split_assets.py --list > "$paths"
{
    printf 'ARG SOURCE_IMAGE=scratch\nFROM ${SOURCE_IMAGE} AS source\nFROM source AS builder\n'
    printf 'COPY docker/split_assets.py docker/prune_cpu.py docker/optimize_libero.py /tmp/\n'
    printf 'RUN if [ "$VLA_IMAGE_PROFILE" = cpu ]; then python /tmp/optimize_libero.py && python /tmp/prune_cpu.py; fi \\\n'
    printf '    && python /tmp/split_assets.py --source-id %s && rm /tmp/split_assets.py /tmp/prune_cpu.py /tmp/optimize_libero.py\n' "$source_id"
    printf 'FROM scratch AS runtime\nCOPY --from=builder / /\n'
    docker image inspect "$source_id" --format '{{range .Config.Env}}{{printf "%q\n" .}}{{end}}' \
        | sed -E 's/^"([^=]+)=(.*)"$/ENV \1="\2"/'
    docker image inspect "$source_id" --format '{{if .Config.WorkingDir}}WORKDIR {{printf "%q" .Config.WorkingDir}}
{{end}}{{if .Config.User}}USER {{printf "%q" .Config.User}}
{{end}}{{range $key, $value := .Config.Labels}}LABEL {{printf "%s=%q" $key $value}}
{{end}}{{if .Config.Entrypoint}}ENTRYPOINT {{json .Config.Entrypoint}}
{{end}}{{if .Config.Cmd}}CMD {{json .Config.Cmd}}{{end}}'
    printf '\nFROM scratch AS assets\n'
    while IFS= read -r path; do
        [[ -z "$path" ]] || printf 'COPY --from=source ["%s", "%s"]\n' "$path" "$path"
    done < "$paths"
    printf 'COPY --from=builder /usr/local/share/vla-build/assets.json /usr/local/share/vla-build/assets.json\n'
} > "$recipe"
# Content-specific temporary tags pin local inputs; BuildKit cannot use bare image IDs in FROM.
profile=$(docker image inspect "$source_id" --format '{{range .Config.Env}}{{println .}}{{end}}' | sed -n 's/^VLA_IMAGE_PROFILE=//p')
[[ "$profile" == cpu || "$profile" == gpu ]] || { echo "Source has no CPU/GPU profile" >&2; exit 1; }
for target in runtime assets; do
    DOCKER_BUILDKIT=1 docker build --network none --target "$target" -f "$recipe" \
        --build-arg "SOURCE_IMAGE=$source_pin" -t "${output}-${target}" .
done
