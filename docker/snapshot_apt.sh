#!/bin/sh
set -eu
snapshot="${1:?UTC snapshot required}"
# Explicit snapshot URLs also work with older apt in CUDA/Ubuntu 22.04 bases.
# The archive is a fallback: the pin prefers snapshot versions whenever its index
# downloads, so an unreachable snapshot.ubuntu.com no longer fails the build.
cat > /etc/apt/sources.list <<EOF
deb [check-valid-until=no] https://snapshot.ubuntu.com/ubuntu/${snapshot}/ jammy main restricted universe multiverse
deb [check-valid-until=no] https://snapshot.ubuntu.com/ubuntu/${snapshot}/ jammy-updates main restricted universe multiverse
deb [check-valid-until=no] https://snapshot.ubuntu.com/ubuntu/${snapshot}/ jammy-security main restricted universe multiverse
deb http://archive.ubuntu.com/ubuntu/ jammy main restricted universe multiverse
deb http://archive.ubuntu.com/ubuntu/ jammy-updates main restricted universe multiverse
deb http://security.ubuntu.com/ubuntu/ jammy-security main restricted universe multiverse
EOF
printf 'Package: *\nPin: origin "snapshot.ubuntu.com"\nPin-Priority: 990\n' > /etc/apt/preferences.d/snapshot
echo 'Acquire::Retries "10";' > /etc/apt/apt.conf.d/80-retries
mkdir -p /usr/local/share/vla-build
printf '%s\n' "$snapshot" > /usr/local/share/vla-build/ubuntu-snapshot.txt
