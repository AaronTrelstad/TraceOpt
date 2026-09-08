#!/bin/bash
# =============================================================================
# run_workloads.sh — TraceOpt Milestone 4: Real workload profiling
#
# Submit from /home/trelstad/Research/TraceOpt:
#   sbatch experiments/real_workloads/run_workloads.sh
# =============================================================================

#SBATCH --job-name=traceopt_m4
#SBATCH --output=logs/m4_%j.out
#SBATCH --error=logs/m4_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-4:00:0
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

echo "=== TraceOpt Milestone 4: Real Workload Profiling ==="
echo "Node: $SLURMD_NODENAME"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "CUDA: $(nvcc --version 2>/dev/null | grep release)"
echo "Python: $(python3 --version)"
echo "Start: $(date)"
echo ""

# ---------------------------------------------------------------------------
# Install dependencies (if needed)
# ---------------------------------------------------------------------------
echo "--- Checking dependencies ---"
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "PyTorch not found. Installing..."
    pip install --user torch torchvision
fi

python3 -c "import torchvision; print(f'torchvision {torchvision.__version__}')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "torchvision not found. Installing..."
    pip install --user torchvision
fi

echo ""

# ---------------------------------------------------------------------------
# Run workloads
# ---------------------------------------------------------------------------
echo "--- Running all workloads ---"
echo ""

cd experiments/real_workloads

# Run each workload individually first (better error isolation)
echo "=== Workload 1: Inference + Logging ==="
python3 workload_inference_logging.py \
    --model resnet18 --batch-size 8 --batches 5 \
    --save-trace trace_inference.json
echo ""

echo "=== Workload 2: Training + GradScaler ==="
python3 workload_training_grad_accum.py \
    --model resnet18 --batch-size 8 --steps 8 --accum 4 \
    --save-trace trace_training.json
echo ""

echo "=== Workload 3: Multi-Stream Pipeline ==="
python3 workload_multi_stream_pipeline.py \
    --model resnet18 --batch-size 8 --batches 6 \
    --save-trace trace_pipeline.json
echo ""

# Run combined analysis (with dependency frontier)
echo "=== Combined Analysis ==="
python3 run_all_workloads.py --output workload_results.json
echo ""

# ---------------------------------------------------------------------------
# DECISIVE EXPERIMENT: Rewrite + measure
# Run as separate python processes to get clean GPU memory
# ---------------------------------------------------------------------------
echo "=== Rewrite Experiment: Baseline vs TraceOpt (bs=16) ==="
echo "(Multi-stream pipeline: DEVICE_SYNC → targeted events)"
echo ""
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null
python3 rewrite_experiment.py \
    --batch-size 16 --iters 20 --trials 30 --warmup 10 \
    --output rewrite_results.json
echo ""

echo "=== Rewrite Experiment: batch=32 ==="
python3 rewrite_experiment.py \
    --batch-size 32 --iters 20 --trials 30 --warmup 10 \
    --output rewrite_results_bs32.json
echo ""

# ---------------------------------------------------------------------------
# SWEEP EXPERIMENT: batch size × execution mode (GLOBAL/EVENT/MANUAL/NONE)
# The decisive experiment for understanding when/why TraceOpt helps
# ---------------------------------------------------------------------------
echo "=== Batch Size Sweep (with MANUAL double-buffered mode) ==="
echo "(Tests: GLOBAL, EVENT, MANUAL, NONE across batch sizes)"
echo ""
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null
python3 rewrite_sweep.py \
    --batch-sizes 1,2,4,8,16,32 \
    --iters 20 --trials 20 --warmup 10 \
    --verify \
    --output sweep_results.json
echo ""

# Try batch=64 separately (may OOM)
echo "=== Batch Size Sweep: bs=64 (may OOM) ==="
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null
python3 rewrite_sweep.py \
    --batch-sizes 64 \
    --iters 10 --trials 15 --warmup 5 \
    --output sweep_results_bs64.json
echo ""

# ---------------------------------------------------------------------------
# PIPELINE DEPTH SWEEP
# ---------------------------------------------------------------------------
echo "=== Pipeline Depth Sweep (bs=16, 2-6 stages) ==="
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null
python3 pipeline_sweep.py \
    --batch-size 16 --stages 2,3,4,5,6 \
    --iters 20 --trials 20 --warmup 10 \
    --output pipeline_sweep_bs16.json
echo ""

echo "=== Pipeline Depth Sweep (bs=8, 2-6 stages) ==="
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null
python3 pipeline_sweep.py \
    --batch-size 8 --stages 2,3,4,5,6 \
    --iters 20 --trials 20 --warmup 10 \
    --output pipeline_sweep_bs8.json
echo ""

# ---------------------------------------------------------------------------
# MILESTONE 5: torch.compile COMPARISON (the kill test)
# ---------------------------------------------------------------------------
echo "=== torch.compile Comparison: 3-stage, bs=16 ==="
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null
python3 compile_comparison.py \
    --batch-size 16 --stages 3 \
    --iters 20 --trials 30 --warmup 15 \
    --output compile_comparison_3s_bs16.json
echo ""

echo "=== torch.compile Comparison: 4-stage, bs=16 ==="
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null
python3 compile_comparison.py \
    --batch-size 16 --stages 4 \
    --iters 20 --trials 30 --warmup 15 \
    --output compile_comparison_4s_bs16.json
echo ""

echo "=== torch.compile Comparison: 3-stage, bs=8 ==="
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null
python3 compile_comparison.py \
    --batch-size 8 --stages 3 \
    --iters 20 --trials 30 --warmup 15 \
    --output compile_comparison_3s_bs8.json
echo ""

# ---------------------------------------------------------------------------
# torch.compile DIAGNOSTIC (resolving the 41→7ms anomaly)
# ---------------------------------------------------------------------------
echo "=== torch.compile Diagnostic ==="
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null
python3 compile_diagnostic.py \
    --batch-size 16 \
    --output compile_diagnostic.json
echo ""

# ---------------------------------------------------------------------------
# NSIGHT SYSTEMS CAPTURE (4 configurations)
# ---------------------------------------------------------------------------
echo "=== Nsight Systems Capture ==="
which nsys > /dev/null 2>&1
if [ $? -eq 0 ]; then
    for MODE in eager_global eager_event compile_global compile_only; do
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
    echo "Nsight profiles saved."
else
    echo "nsys not found, skipping Nsight capture."
fi
echo ""

echo "End: $(date)"
echo "Exit: $?"
