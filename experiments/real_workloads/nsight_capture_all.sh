#!/bin/bash
# Capture Nsight Systems profiles for 4 key configurations.
# Run from experiments/real_workloads/
#
# Output: .nsys-rep files that can be opened in Nsight Systems GUI
# or analyzed with: nsys stats <file>.nsys-rep

set -e

BS=16
ITERS=10   # Keep small for readable timelines
WARMUP=5

echo "=== Nsight Systems Capture ==="
echo "Batch size: $BS, Iters: $ITERS"
echo ""

for MODE in eager_global eager_event compile_global compile_only; do
    echo "--- Capturing: $MODE ---"
    nsys profile \
        --trace=cuda,nvtx \
        --cuda-memory-usage=true \
        --output="nsight_${MODE}" \
        --force-overwrite=true \
        python3 nsight_capture.py \
            --mode $MODE \
            --batch-size $BS \
            --iters $ITERS \
            --warmup $WARMUP
    echo ""
done

echo "=== Profiles saved ==="
ls -la nsight_*.nsys-rep 2>/dev/null || echo "(no .nsys-rep files found)"

echo ""
echo "To view summaries:"
echo "  nsys stats nsight_eager_global.nsys-rep"
echo "  nsys stats nsight_compile_global.nsys-rep"
echo ""
echo "To compare GPU active time:"
echo "  nsys stats --report cuda_gpu_kern_sum nsight_eager_global.nsys-rep"
echo "  nsys stats --report cuda_gpu_kern_sum nsight_compile_only.nsys-rep"
