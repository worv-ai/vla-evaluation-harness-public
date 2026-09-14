#!/bin/sh
set -eu
if [ "${VLA_IMAGE_PROFILE:-gpu}" = cpu ]; then
    case "${2:-}" in
        run|test) set -- "$@" --render cpu ;;
    esac
fi
exec "$@"
