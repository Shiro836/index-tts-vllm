#!/bin/bash
# Block until the GPU has at least $1 MiB of free VRAM (default 12000),
# giving up after $2 seconds (default 300).
#
# Used as ExecStartPre so the service doesn't race other GPU services at
# boot, or its own previous instance whose EngineCore processes are still
# releasing memory after a crash/restart.

REQUIRED_MB="${1:-12000}"
TIMEOUT_S="${2:-300}"

deadline=$(( $(date +%s) + TIMEOUT_S ))
while true; do
    free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [[ "$free_mb" =~ ^[0-9]+$ ]] && (( free_mb >= REQUIRED_MB )); then
        echo "wait_for_vram: ${free_mb}MiB free (>= ${REQUIRED_MB}MiB), proceeding"
        exit 0
    fi
    if (( $(date +%s) >= deadline )); then
        echo "wait_for_vram: timed out after ${TIMEOUT_S}s (free: ${free_mb:-unknown}MiB, need ${REQUIRED_MB}MiB)" >&2
        exit 1
    fi
    sleep 5
done
