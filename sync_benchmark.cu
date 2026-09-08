/**
 * TraceOpt: Synchronization Weakening Performance Benchmark
 *
 * Measures the critical-path impact of replacing a global
 * cudaDeviceSynchronize() with targeted CUDA event dependencies.
 *
 * Topology:
 *   Stream 0: A ─── [barrier] ─── B
 *   Stream 1: C  (independent, blocked by global barrier)
 *   Stream 2: D  (independent, blocked by global barrier)
 *   ...up to N streams
 *
 * Three sync modes:
 *   GLOBAL:  cudaDeviceSynchronize() between phases
 *   EVENT:   cudaEventRecord/Wait only for the required A→B dependency
 *   NONE:    no synchronization (unsafe baseline, shows max overlap)
 *
 * Usage:
 *   ./sync_benchmark --streams 4 --t_a 4.0 --t_b 4.0 --t_c 4.0 --trials 100
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

// Busy-wait kernel: spins for approximately `duration_ms` milliseconds.
// Uses clock() for device-side timing.
__global__ void busy_kernel(float duration_ms, float *out, int n) {
    long long start = clock64();
    long long target = (long long)(duration_ms * 1e-3f *
                                    __ldg(&out[0]) * 0.0f);  // prevent optimization
    // Use the SM clock frequency to spin.
    // clock64() ticks at the SM clock rate.
    // We calibrate by doing a short spin first.
    float elapsed_ms = 0.0f;

    // Approximate: SM clock is ~1.5 GHz on modern GPUs.
    // We'll use a self-calibrating approach.
    long long freq_estimate = 1500000000LL;  // 1.5 GHz default

    long long ticks_needed = (long long)(duration_ms * 1e-3 * freq_estimate);

    while (clock64() - start < ticks_needed) {
        // spin
    }

    // Write something to prevent dead-code elimination
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out[0] = (float)(clock64() - start);
    }
}

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

struct Config {
    int num_streams;       // Total streams (1 = stream 0 only)
    float t_a_ms;          // Duration of kernel A on stream 0 (before barrier)
    float t_b_ms;          // Duration of kernel B on stream 0 (after barrier)
    float t_c_ms;          // Duration of independent kernels on other streams
    int trials;            // Number of repetitions
    int warmup;            // Warmup iterations
};

enum SyncMode { GLOBAL, EVENT, NONE };
const char *sync_mode_name[] = {"GLOBAL", "EVENT", "NONE"};

float run_trial(Config *cfg, SyncMode mode, cudaStream_t *streams,
                cudaEvent_t event_a, float **d_out) {
    cudaEvent_t start_ev, end_ev;
    CHECK_CUDA(cudaEventCreate(&start_ev));
    CHECK_CUDA(cudaEventCreate(&end_ev));

    CHECK_CUDA(cudaEventRecord(start_ev, 0));

    // Phase 1: Launch A on stream 0, C/D/... on other streams
    busy_kernel<<<1, 1, 0, streams[0]>>>(cfg->t_a_ms, d_out[0], 1);

    for (int i = 1; i < cfg->num_streams; i++) {
        busy_kernel<<<1, 1, 0, streams[i]>>>(cfg->t_c_ms, d_out[i], 1);
    }

    // Synchronization
    switch (mode) {
        case GLOBAL:
            // Global barrier: waits for ALL streams
            CHECK_CUDA(cudaDeviceSynchronize());
            break;

        case EVENT:
            // Targeted event: only A→B dependency on stream 0
            CHECK_CUDA(cudaEventRecord(event_a, streams[0]));
            CHECK_CUDA(cudaStreamWaitEvent(streams[0], event_a, 0));
            // Other streams are NOT blocked
            break;

        case NONE:
            // No sync at all (unsafe, but shows max overlap)
            break;
    }

    // Phase 2: Launch B on stream 0 (depends on A)
    busy_kernel<<<1, 1, 0, streams[0]>>>(cfg->t_b_ms, d_out[0], 1);

    CHECK_CUDA(cudaEventRecord(end_ev, 0));

    // Wait for everything to finish for timing
    CHECK_CUDA(cudaDeviceSynchronize());

    float elapsed_ms;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start_ev, end_ev));

    CHECK_CUDA(cudaEventDestroy(start_ev));
    CHECK_CUDA(cudaEventDestroy(end_ev));

    return elapsed_ms;
}

void run_experiment(Config *cfg) {
    printf("=== TraceOpt Sync Weakening Benchmark ===\n");
    printf("Streams: %d\n", cfg->num_streams);
    printf("T_A: %.1f ms, T_B: %.1f ms, T_C: %.1f ms\n",
           cfg->t_a_ms, cfg->t_b_ms, cfg->t_c_ms);
    printf("Trials: %d (warmup: %d)\n", cfg->trials, cfg->warmup);

    // Print GPU info
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (SM %d.%d, %d SMs)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    printf("Clock: %d MHz\n\n", prop.clockRate / 1000);

    // Create streams
    cudaStream_t *streams = (cudaStream_t *)malloc(
        cfg->num_streams * sizeof(cudaStream_t));
    for (int i = 0; i < cfg->num_streams; i++) {
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
    }

    // Create event for A→B dependency
    cudaEvent_t event_a;
    CHECK_CUDA(cudaEventCreate(&event_a));

    // Allocate device memory (one per stream, to prevent false sharing)
    float **d_out = (float **)malloc(cfg->num_streams * sizeof(float *));
    for (int i = 0; i < cfg->num_streams; i++) {
        CHECK_CUDA(cudaMalloc(&d_out[i], sizeof(float)));
        CHECK_CUDA(cudaMemset(d_out[i], 0, sizeof(float)));
    }

    // Run for each sync mode
    for (int mode = 0; mode <= 2; mode++) {
        SyncMode sm = (SyncMode)mode;

        // Warmup
        for (int i = 0; i < cfg->warmup; i++) {
            run_trial(cfg, sm, streams, event_a, d_out);
        }

        // Collect measurements
        float total = 0.0f;
        float min_t = FLT_MAX;
        float max_t = 0.0f;
        float *times = (float *)malloc(cfg->trials * sizeof(float));

        for (int i = 0; i < cfg->trials; i++) {
            times[i] = run_trial(cfg, sm, streams, event_a, d_out);
            total += times[i];
            if (times[i] < min_t) min_t = times[i];
            if (times[i] > max_t) max_t = times[i];
        }

        float mean = total / cfg->trials;

        // Compute std dev
        float var = 0.0f;
        for (int i = 0; i < cfg->trials; i++) {
            float diff = times[i] - mean;
            var += diff * diff;
        }
        float stddev = sqrtf(var / cfg->trials);

        // Sort for median/p95/p99
        for (int i = 0; i < cfg->trials - 1; i++) {
            for (int j = i + 1; j < cfg->trials; j++) {
                if (times[j] < times[i]) {
                    float tmp = times[i];
                    times[i] = times[j];
                    times[j] = tmp;
                }
            }
        }
        float median = times[cfg->trials / 2];
        float p95 = times[(int)(cfg->trials * 0.95)];
        float p99 = times[(int)(cfg->trials * 0.99)];

        printf("%-8s  mean=%.3f ms  median=%.3f ms  std=%.3f ms  "
               "min=%.3f ms  max=%.3f ms  p95=%.3f ms  p99=%.3f ms\n",
               sync_mode_name[mode], mean, median, stddev,
               min_t, max_t, p95, p99);

        free(times);
    }

    // Cleanup
    for (int i = 0; i < cfg->num_streams; i++) {
        CHECK_CUDA(cudaFree(d_out[i]));
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
    }
    CHECK_CUDA(cudaEventDestroy(event_a));
    free(d_out);
    free(streams);
}

void print_usage(const char *prog) {
    fprintf(stderr, "Usage: %s [options]\n", prog);
    fprintf(stderr, "  --streams N    Number of streams (default: 2)\n");
    fprintf(stderr, "  --t_a MS       Kernel A duration in ms (default: 4.0)\n");
    fprintf(stderr, "  --t_b MS       Kernel B duration in ms (default: 4.0)\n");
    fprintf(stderr, "  --t_c MS       Independent kernel duration (default: 4.0)\n");
    fprintf(stderr, "  --trials N     Number of trials (default: 100)\n");
    fprintf(stderr, "  --warmup N     Warmup iterations (default: 10)\n");
}

int main(int argc, char **argv) {
    Config cfg = {
        .num_streams = 2,
        .t_a_ms = 4.0f,
        .t_b_ms = 4.0f,
        .t_c_ms = 4.0f,
        .trials = 100,
        .warmup = 10,
    };

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--streams") == 0 && i + 1 < argc)
            cfg.num_streams = atoi(argv[++i]);
        else if (strcmp(argv[i], "--t_a") == 0 && i + 1 < argc)
            cfg.t_a_ms = atof(argv[++i]);
        else if (strcmp(argv[i], "--t_b") == 0 && i + 1 < argc)
            cfg.t_b_ms = atof(argv[++i]);
        else if (strcmp(argv[i], "--t_c") == 0 && i + 1 < argc)
            cfg.t_c_ms = atof(argv[++i]);
        else if (strcmp(argv[i], "--trials") == 0 && i + 1 < argc)
            cfg.trials = atoi(argv[++i]);
        else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc)
            cfg.warmup = atoi(argv[++i]);
        else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (cfg.num_streams < 1) {
        fprintf(stderr, "Need at least 1 stream\n");
        return 1;
    }

    run_experiment(&cfg);

    // Run sweep: vary number of streams
    printf("\n=== Stream Count Sweep ===\n");
    printf("(T_A=%.1f, T_B=%.1f, T_C=%.1f)\n\n",
           cfg.t_a_ms, cfg.t_b_ms, cfg.t_c_ms);

    int stream_counts[] = {1, 2, 4, 8};
    int n_counts = 4;

    printf("%-8s", "Streams");
    for (int mode = 0; mode <= 2; mode++) {
        printf("  %-12s", sync_mode_name[mode]);
    }
    printf("  Speedup(E/G)\n");

    for (int si = 0; si < n_counts; si++) {
        Config sweep_cfg = cfg;
        sweep_cfg.num_streams = stream_counts[si];
        sweep_cfg.trials = 50;
        sweep_cfg.warmup = 5;

        // Create resources
        cudaStream_t *streams = (cudaStream_t *)malloc(
            sweep_cfg.num_streams * sizeof(cudaStream_t));
        for (int i = 0; i < sweep_cfg.num_streams; i++) {
            CHECK_CUDA(cudaStreamCreate(&streams[i]));
        }
        cudaEvent_t event_a;
        CHECK_CUDA(cudaEventCreate(&event_a));
        float **d_out = (float **)malloc(
            sweep_cfg.num_streams * sizeof(float *));
        for (int i = 0; i < sweep_cfg.num_streams; i++) {
            CHECK_CUDA(cudaMalloc(&d_out[i], sizeof(float)));
            CHECK_CUDA(cudaMemset(d_out[i], 0, sizeof(float)));
        }

        float means[3];
        for (int mode = 0; mode <= 2; mode++) {
            SyncMode sm = (SyncMode)mode;

            // Warmup
            for (int w = 0; w < sweep_cfg.warmup; w++) {
                run_trial(&sweep_cfg, sm, streams, event_a, d_out);
            }

            float total = 0.0f;
            for (int t = 0; t < sweep_cfg.trials; t++) {
                total += run_trial(&sweep_cfg, sm, streams, event_a, d_out);
            }
            means[mode] = total / sweep_cfg.trials;
        }

        float speedup = (means[1] > 0) ? means[0] / means[1] : 0.0f;

        printf("%-8d  %-12.3f  %-12.3f  %-12.3f  %.3fx\n",
               stream_counts[si], means[0], means[1], means[2], speedup);

        // Cleanup
        for (int i = 0; i < sweep_cfg.num_streams; i++) {
            CHECK_CUDA(cudaFree(d_out[i]));
            CHECK_CUDA(cudaStreamDestroy(streams[i]));
        }
        CHECK_CUDA(cudaEventDestroy(event_a));
        free(d_out);
        free(streams);
    }

    // Run sweep: vary T_C (independent work duration)
    printf("\n=== Independent Work Duration Sweep ===\n");
    printf("(Streams=%d, T_A=%.1f, T_B=%.1f)\n\n",
           cfg.num_streams, cfg.t_a_ms, cfg.t_b_ms);

    float tc_values[] = {0.5f, 1.0f, 2.0f, 4.0f, 8.0f, 16.0f};
    int n_tc = 6;

    printf("%-8s", "T_C(ms)");
    for (int mode = 0; mode <= 2; mode++) {
        printf("  %-12s", sync_mode_name[mode]);
    }
    printf("  Speedup(E/G)\n");

    for (int ti = 0; ti < n_tc; ti++) {
        Config sweep_cfg = cfg;
        sweep_cfg.t_c_ms = tc_values[ti];
        sweep_cfg.trials = 50;
        sweep_cfg.warmup = 5;

        cudaStream_t *streams = (cudaStream_t *)malloc(
            sweep_cfg.num_streams * sizeof(cudaStream_t));
        for (int i = 0; i < sweep_cfg.num_streams; i++) {
            CHECK_CUDA(cudaStreamCreate(&streams[i]));
        }
        cudaEvent_t event_a;
        CHECK_CUDA(cudaEventCreate(&event_a));
        float **d_out = (float **)malloc(
            sweep_cfg.num_streams * sizeof(float *));
        for (int i = 0; i < sweep_cfg.num_streams; i++) {
            CHECK_CUDA(cudaMalloc(&d_out[i], sizeof(float)));
            CHECK_CUDA(cudaMemset(d_out[i], 0, sizeof(float)));
        }

        float means[3];
        for (int mode = 0; mode <= 2; mode++) {
            SyncMode sm = (SyncMode)mode;
            for (int w = 0; w < sweep_cfg.warmup; w++) {
                run_trial(&sweep_cfg, sm, streams, event_a, d_out);
            }
            float total = 0.0f;
            for (int t = 0; t < sweep_cfg.trials; t++) {
                total += run_trial(&sweep_cfg, sm, streams, event_a, d_out);
            }
            means[mode] = total / sweep_cfg.trials;
        }

        float speedup = (means[1] > 0) ? means[0] / means[1] : 0.0f;

        printf("%-8.1f  %-12.3f  %-12.3f  %-12.3f  %.3fx\n",
               tc_values[ti], means[0], means[1], means[2], speedup);

        for (int i = 0; i < sweep_cfg.num_streams; i++) {
            CHECK_CUDA(cudaFree(d_out[i]));
            CHECK_CUDA(cudaStreamDestroy(streams[i]));
        }
        CHECK_CUDA(cudaEventDestroy(event_a));
        free(d_out);
        free(streams);
    }

    return 0;
}
