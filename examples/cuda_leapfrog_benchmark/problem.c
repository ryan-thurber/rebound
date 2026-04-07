/**
 * CUDA Leapfrog Benchmark
 * Tests kick/drift only — gravity excluded from timing
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "rebound.h"
#include "integrator.h"

// ─── Timing ──────────────────────────────────────────────────────────────────

static double wall_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// ─── Setup ───────────────────────────────────────────────────────────────────

static void setup_simulation(struct reb_simulation* r, int N) {
    r->dt      = 0.01;
    r->gravity = REB_GRAVITY_NONE;

    reb_simulation_add_fmt(r, "m", 1.0);

    srand(42);
    for (int i = 1; i < N; i++) {
        struct reb_particle p = {0};
        p.m  = 1e-6;
        p.x  = ((double)rand() / RAND_MAX) * 10.0 - 5.0;
        p.y  = ((double)rand() / RAND_MAX) * 10.0 - 5.0;
        p.z  = ((double)rand() / RAND_MAX) * 2.0  - 1.0;
        p.vx = ((double)rand() / RAND_MAX) * 2.0  - 1.0;
        p.vy = ((double)rand() / RAND_MAX) * 2.0  - 1.0;
        p.vz = ((double)rand() / RAND_MAX) * 0.2  - 0.1;
        p.ax = ((double)rand() / RAND_MAX) * 0.01;
        p.ay = ((double)rand() / RAND_MAX) * 0.01;
        p.az = ((double)rand() / RAND_MAX) * 0.001;
        reb_simulation_add(r, p);
    }
}

// ─── Single run ──────────────────────────────────────────────────────────────

typedef struct {
    double elapsed;
    double energy_error;
} BenchResult;

// Declared in integrator_leapfrog_cuda.cu
extern void reb_leapfrog_cuda_kick_drift_only(struct reb_simulation* r, int n_steps);

static BenchResult run_bench(int N, int integrator, int n_steps) {
    struct reb_simulation* r = reb_simulation_create();
    r->integrator = integrator;
    setup_simulation(r, N);

    // Pre-compute accelerations once — not part of timing
    reb_simulation_update_acceleration(r);
    double E0 = reb_simulation_energy(r);

    double t0 = wall_time();

    if (integrator == REB_INTEGRATOR_LEAPFROG) {
        for (int s = 0; s < n_steps; s++) {
            // drift
            for (int i = 0; i < N; i++) {
                r->particles[i].x  += r->dt * r->particles[i].vx;
                r->particles[i].y  += r->dt * r->particles[i].vy;
                r->particles[i].z  += r->dt * r->particles[i].vz;
            }
            // kick
            for (int i = 0; i < N; i++) {
                r->particles[i].vx += r->dt * r->particles[i].ax;
                r->particles[i].vy += r->dt * r->particles[i].ay;
                r->particles[i].vz += r->dt * r->particles[i].az;
            }
        }
    } else {
        reb_leapfrog_cuda_kick_drift_only(r, n_steps);
    }

    double elapsed = wall_time() - t0;
    double dE = fabs((reb_simulation_energy(r) - E0) / E0);

    reb_simulation_free(r);
    BenchResult res = { elapsed, dE };
    return res;
}

// ─── Main ────────────────────────────────────────────────────────────────────

int main() {
    int N_values[] = { 100, 1000, 10000, 100000, 500000 };
    int n_N        = sizeof(N_values) / sizeof(N_values[0]);
    int n_steps    = 1000;

    struct {
        const char* name;
        int type;
    } integrators[] = {
        { "Leapfrog (serial)", REB_INTEGRATOR_LEAPFROG      },
        { "Leapfrog (CUDA)",   REB_INTEGRATOR_LEAPFROG_CUDA },
    };
    int n_integrators = sizeof(integrators) / sizeof(integrators[0]);

    double serial_times[5] = {0};

    printf("\nKick/Drift Benchmark (gravity excluded) | %d steps per run\n\n", n_steps);
    printf("%-8s | %-22s | %-12s | %-12s | %s\n",
           "N", "Integrator", "Time (s)", "|dE/E|", "Speedup");
    printf("─────────────────────────────────────────────────────────────────\n");

    for (int i = 0; i < n_N; i++) {
        int N = N_values[i];
        for (int j = 0; j < n_integrators; j++) {
            BenchResult res = run_bench(N, integrators[j].type, n_steps);

            double speedup = 1.0;
            if (integrators[j].type == REB_INTEGRATOR_LEAPFROG) {
                serial_times[i] = res.elapsed;
            } else {
                speedup = serial_times[i] / res.elapsed;
            }

            printf("%-8d | %-22s | %-12.4f | %-12.2e | %.2fx\n",
                   N, integrators[j].name,
                   res.elapsed, res.energy_error, speedup);
        }
    }

    printf("─────────────────────────────────────────────────────────────────\n");
    return 0;
}
