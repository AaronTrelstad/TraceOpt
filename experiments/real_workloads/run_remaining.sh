#!/bin/bash
# =============================================================================
# run_remaining.sh — Only run experiments not yet completed
#
# Submit from /home/trelstad/Research/TraceOpt:
#   sbatch experiments/real_workloads/run_remaining.sh
# =============================================================================

#SBATCH --job-name=traceopt_remaining
#SBATCH --output=logs/remaining_%j.out
#SBATCH --error=logs/remaining_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-2:00:0
#SBATCH --partition=instruction
#SBATCH --gres=gpu:1
#SBATCH --account=f2026.coms.5790.01
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=trelstad@iastate.edu

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module purge
module load cuda/12.6.3-la3bxnl

cd $SLURM_SUBMIT_DIR
mkdir -p logs

export CUDA_VISIBLE_DEVICES=0

echo "=== TraceOpt: Remaining Experiments ==="
echo "Node: $SLURMD_NODENAME"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Start: $(date)"
echo ""

cd experiments/real_workloads

# ---------------------------------------------------------------------------
# 1. EAGER TIMING (focused, 40 trials, high confidence)
# ---------------------------------------------------------------------------
echo "=== Eager Timing Experiment ==="
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null
python3 eager_timing.py \
    --output eager_timing.json
echo ""

# ---------------------------------------------------------------------------
# 2. NSIGHT CAPTURES: compile_RO_global vs compile_RO_event
# ---------------------------------------------------------------------------
echo "=== Nsight: compile_RO_global vs compile_RO_event ==="
which nsys > /dev/null 2>&1
if [ $? -eq 0 ]; then
    for MODE in compile_RO_global compile_RO_event; do
        echo "--- Capturing: $MODE ---"
        nsys profile \
            --trace=cuda,nvtx \
            --output="nsight_${MODE}" \
            --force-overwrite=true \
            python3 nsight_capture.py \
                --mode $MODE \
                --batch-size 16 \
                --iters 10 \
                --warmup 5
        echo ""
    done
    bash nsight_analyze.sh 2>/dev/null || true
else
    echo "nsys not found, skipping Nsight capture."
fi
echo ""

echo "End: $(date)"
echo "Exit: $?"
