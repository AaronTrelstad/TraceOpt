#!/bin/bash
# =============================================================================
# run_benchmark.sh — TraceOpt sync weakening performance benchmark
#
# Submit:
#   sbatch experiments/benchmark/run_benchmark.sh
# =============================================================================

#SBATCH --job-name=traceopt_bench
#SBATCH --output=experiments/benchmark/logs/%j.out
#SBATCH --error=experiments/benchmark/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-0:30:0
#SBATCH --partition=instruction
#SBATCH --gres=gpu:1
#SBATCH --account=f2026.coms.5790.01
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=trelstad@iastate.edu

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module purge
module load cuda    # Adjust if your cluster uses a different module name

cd $SLURM_SUBMIT_DIR

mkdir -p experiments/benchmark/logs

export CUDA_VISIBLE_DEVICES=0

echo "=== TraceOpt Sync Weakening Benchmark ==="
echo "Node: $SLURMD_NODENAME"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "CUDA: $(nvcc --version 2>/dev/null | grep release)"
echo "Start: $(date)"
echo ""

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
echo "--- Building benchmark ---"

# Detect GPU architecture
GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
if [ -z "$GPU_ARCH" ]; then
    GPU_ARCH=70
fi
echo "Detected compute capability: sm_${GPU_ARCH}"

cd experiments/benchmark
make clean
NVCC_FLAGS="-O2 -arch=sm_${GPU_ARCH}" make
cd $SLURM_SUBMIT_DIR

echo ""

# ---------------------------------------------------------------------------
# Run: Default configuration
# ---------------------------------------------------------------------------
echo "--- Experiment 1: Default (2 streams, 4ms each) ---"
./experiments/benchmark/sync_benchmark \
    --streams 2 --t_a 4.0 --t_b 4.0 --t_c 4.0 --trials 100 --warmup 20

echo ""

# ---------------------------------------------------------------------------
# Run: Asymmetric durations
# ---------------------------------------------------------------------------
echo "--- Experiment 2: Short A, long C (A=1ms, B=4ms, C=8ms) ---"
./experiments/benchmark/sync_benchmark \
    --streams 2 --t_a 1.0 --t_b 4.0 --t_c 8.0 --trials 100 --warmup 20

echo ""

echo "--- Experiment 3: Long A, short C (A=8ms, B=4ms, C=1ms) ---"
./experiments/benchmark/sync_benchmark \
    --streams 2 --t_a 8.0 --t_b 4.0 --t_c 1.0 --trials 100 --warmup 20

echo ""

# ---------------------------------------------------------------------------
# Run: Many streams
# ---------------------------------------------------------------------------
echo "--- Experiment 4: 4 streams, equal durations ---"
./experiments/benchmark/sync_benchmark \
    --streams 4 --t_a 4.0 --t_b 4.0 --t_c 4.0 --trials 100 --warmup 20

echo ""

echo "--- Experiment 5: 8 streams, equal durations ---"
./experiments/benchmark/sync_benchmark \
    --streams 8 --t_a 4.0 --t_b 4.0 --t_c 4.0 --trials 100 --warmup 20

echo ""

# ---------------------------------------------------------------------------
# Run: Realistic ML-like durations
# ---------------------------------------------------------------------------
echo "--- Experiment 6: ML-like (A=2ms attn, B=1ms ffn, C=3ms preprocess) ---"
./experiments/benchmark/sync_benchmark \
    --streams 2 --t_a 2.0 --t_b 1.0 --t_c 3.0 --trials 100 --warmup 20

echo ""

echo "End: $(date)"
echo "Exit: $?"
