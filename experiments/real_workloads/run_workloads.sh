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
#SBATCH --time=0-1:00:0
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

# Run combined analysis
echo "=== Combined Analysis ==="
python3 run_all_workloads.py --output workload_results.json
echo ""

echo "End: $(date)"
echo "Exit: $?"
