/**
 * @file problem.c
 * @brief CUDA Leapfrog Integration Test
 * Uses acceleration values from the serial gravity "basic" calculation
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "rebound.h"
#include "integrator_leapfrog.h"
#include "integrator_leapfrog_cuda.h"

typedef struct {
    double elapsed;
    double energy_error;
} BenchResult;

static double wall_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void setup_simulation(struct reb_simulation* r, int N) {
    r->dt      = 0.001;
    r->gravity = REB_GRAVITY_BASIC;

    struct reb_particle sun = {0};
    sun.m = 1.0;
    reb_simulation_add(r, sun);

 // set random initial conditions for now
    srand(42);
    for (int i = 1; i < N; i++) {
        struct reb_particle p = {0};
        double dist = 1.0 + (double)rand()/RAND_MAX * 5.0;
        double ang  = (double)rand()/RAND_MAX * 2.0 * M_PI;

        p.m  = 1e-9;
        p.x  = dist * cos(ang);
        p.y  = dist * sin(ang);
        p.z  = ((double)rand()/RAND_MAX - 0.5) * 0.1;

        double vmag = sqrt(1.0 / dist);
        p.vx = -vmag * sin(ang);
        p.vy =  vmag * cos(ang);

        reb_simulation_add(r, p);
    }
}

static BenchResult run_bench(int N, int integrator_type, int n_steps) {
    struct reb_simulation* r = reb_simulation_create();
    r->integrator = (integrator_type == REB_INTEGRATOR_LEAPFROG)
                    ? REB_INTEGRATOR_LEAPFROG
                    : REB_INTEGRATOR_NONE;
    setup_simulation(r, N);

    reb_simulation_update_acceleration(r);
    double E0 = reb_simulation_energy(r);

    double t0 = wall_time();

    for (int s = 0; s < n_steps; s++) {
        if (integrator_type == REB_INTEGRATOR_LEAPFROG) {
            reb_integrator_leapfrog_step(r);
        } else {
            reb_integrator_leapfrog_cuda_step(r);
        }
    }

    double elapsed = wall_time() - t0;
    double dE = fabs((reb_simulation_energy(r) - E0) / E0);

    reb_simulation_free(r);
    BenchResult res = { elapsed, dE };
    return res;
}

int main(int argc, char* argv[]) {
    int N_values[] = { 100, 500, 1000, 2000 };
    int n_N        = sizeof(N_values) / sizeof(N_values[0]);  // fixed
    int n_steps    = 100;

    printf("\nReal Gravity Leapfrog Test | %d steps | O(N^2) CPU Gravity\n", n_steps);
    printf("%-8s | %-20s | %-12s | %-12s | %s\n",
           "N", "Integrator", "Time (s)", "|dE/E|", "Speedup");
    printf("----------------------------------------------------------------------\n");

    for (int i = 0; i < n_N; i++) {
        int N = N_values[i];

        BenchResult res_s = run_bench(N, REB_INTEGRATOR_LEAPFROG, n_steps);
        printf("%-8d | Leapfrog (Serial)    | %-12.4f | %-12.2e | 1.00x\n",
               N, res_s.elapsed, res_s.energy_error);

        BenchResult res_c = run_bench(N, REB_INTEGRATOR_NONE, n_steps);
        double speedup = res_s.elapsed / res_c.elapsed;
        printf("%-8d | Leapfrog (CUDA)      | %-12.4f | %-12.2e | %.2fx\n",
               N, res_c.elapsed, res_c.energy_error, speedup);
        printf("----------------------------------------------------------------------\n");
    }

    return 0;
}
