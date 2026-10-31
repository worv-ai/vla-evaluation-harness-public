# Sourced by the profiling scripts: shared-filesystem caches for vla-eval's Charliecloud runtime on this cluster.
U=/mnt/cepheid/users/$USER
export VLA_EVAL_HOME=${VLA_EVAL_HOME:-$U/vla-eval-cache} CH_IMAGE_STORAGE=${CH_IMAGE_STORAGE:-$U/ch-image-storage}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$U/.cache/uv} TMPDIR=${TMPDIR:-$U/tmp}
export PATH=$U/work/squashfuse/.pixi/envs/default/bin:$HOME/.pixi/bin:$PATH  # squashfuse, ch-*
export VLA_EVAL_RUNTIME=charliecloud
mkdir -p "$TMPDIR"
