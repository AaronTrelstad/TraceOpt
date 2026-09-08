#!/bin/bash
# Extract Nsight stats as text files for local analysis.
# Run on the cluster where nsys is available.
#
# Usage: bash nsight_analyze.sh

set -e

OUTDIR="nsight_stats"
mkdir -p "$OUTDIR"

for PROFILE in nsight_eager_global nsight_eager_event nsight_compile_global nsight_compile_only; do
    REP="${PROFILE}.nsys-rep"
    if [ ! -f "$REP" ]; then
        echo "Skipping $REP (not found)"
        continue
    fi

    echo "=== Analyzing: $PROFILE ==="

    # CUDA API summary (shows cudaDeviceSynchronize, cudaStreamWaitEvent, etc.)
    echo "--- cuda_api_sum ---"
    nsys stats --report cuda_api_sum "$REP" > "$OUTDIR/${PROFILE}_cuda_api.txt" 2>&1
    cat "$OUTDIR/${PROFILE}_cuda_api.txt"
    echo ""

    # CUDA kernel summary (shows per-kernel GPU time)
    echo "--- cuda_gpu_kern_sum ---"
    nsys stats --report cuda_gpu_kern_sum "$REP" > "$OUTDIR/${PROFILE}_kernels.txt" 2>&1
    cat "$OUTDIR/${PROFILE}_kernels.txt"
    echo ""

    # NVTX ranges (our annotations: iter_N, preprocess, inference, etc.)
    echo "--- nvtx_sum ---"
    nsys stats --report nvtx_sum "$REP" > "$OUTDIR/${PROFILE}_nvtx.txt" 2>&1
    cat "$OUTDIR/${PROFILE}_nvtx.txt"
    echo ""

    echo "============================================================"
    echo ""
done

# Quick comparison: sync time
echo ""
echo "=== SYNC TIME COMPARISON ==="
echo ""
for PROFILE in nsight_eager_global nsight_compile_global; do
    REP="${PROFILE}.nsys-rep"
    if [ ! -f "$REP" ]; then continue; fi
    echo "--- $PROFILE ---"
    nsys stats --report cuda_api_sum "$REP" 2>/dev/null | grep -iE "synch|Synchronize|Total"
    echo ""
done

echo "=== EVENT TIME COMPARISON ==="
echo ""
for PROFILE in nsight_eager_event nsight_compile_only; do
    REP="${PROFILE}.nsys-rep"
    if [ ! -f "$REP" ]; then continue; fi
    echo "--- $PROFILE ---"
    nsys stats --report cuda_api_sum "$REP" 2>/dev/null | grep -iE "Wait|Record|Event|Graph|Total"
    echo ""
done

echo ""
echo "Stats saved to $OUTDIR/"
echo "Download these files for local review."
