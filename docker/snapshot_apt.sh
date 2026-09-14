#!/bin/sh
set -eu
snapshot="${1:?UTC snapshot required}"
# Explicit snapshot URLs also work with older apt in CUDA/Ubuntu 22.04 bases.
cat > /etc/apt/sources.list <<EOF
deb [check-valid-until=no] https://snapshot.ubuntu.com/ubuntu/${snapshot}/ jammy main restricted universe multiverse
deb [check-valid-until=no] https://snapshot.ubuntu.com/ubuntu/${snapshot}/ jammy-updates main restricted universe multiverse
deb [check-valid-until=no] https://snapshot.ubuntu.com/ubuntu/${snapshot}/ jammy-security main restricted universe multiverse
EOF
mkdir -p /usr/local/share/vla-build
printf '%s\n' "$snapshot" > /usr/local/share/vla-build/ubuntu-snapshot.txt
