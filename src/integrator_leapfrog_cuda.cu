// integrator_leapfrog_cuda.cu

#include <cuda_runtime.h>
extern "C" {
#include "rebound.h"
#include "integrator_leapfrog_cuda.h"
}

// ─── Kernels ─────────────────────────────────────────────────────────────────

__global__ void drift_kernel(struct reb_particle* particles, int N, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    particles[i].x  += dt * particles[i].vx;
    particles[i].y  += dt * particles[i].vy;
    particles[i].z  += dt * particles[i].vz;
}

__global__ void kick_kernel(struct reb_particle* particles, int N, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    particles[i].vx += dt * particles[i].ax;
    particles[i].vy += dt * particles[i].ay;
    particles[i].vz += dt * particles[i].az;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

#define BLOCK_SIZE 256

static void launch_drift(struct reb_particle* d_particles, int N, double dt) {
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    drift_kernel<<<blocks, BLOCK_SIZE>>>(d_particles, N, dt);
}

static void launch_kick(struct reb_particle* d_particles, int N, double dt) {
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kick_kernel<<<blocks, BLOCK_SIZE>>>(d_particles, N, dt);
}

// ─── GPU memory management ────────────────────────────────────────────────────

static struct reb_particle* d_particles = NULL;
static int d_N_allocated = 0;

static void sync_to_gpu(struct reb_simulation* r) {
    int N = r->N;
    if (N > d_N_allocated) {
        if (d_particles) cudaFree(d_particles);
        cudaMalloc(&d_particles, N * sizeof(struct reb_particle));
        d_N_allocated = N;
    }
    cudaMemcpy(d_particles, r->particles, N * sizeof(struct reb_particle),
               cudaMemcpyHostToDevice);
}

static void sync_to_cpu(struct reb_simulation* r) {
    cudaMemcpy(r->particles, d_particles, r->N * sizeof(struct reb_particle),
               cudaMemcpyDeviceToHost);
}

// ─── Kick/drift only (for benchmarking without gravity) ───────────────────────

extern "C" void reb_leapfrog_cuda_kick_drift_only(struct reb_simulation* r, int n_steps) {
    int N = r->N;
    sync_to_gpu(r);
    for (int s = 0; s < n_steps; s++) {
        launch_drift(d_particles, N, r->dt);
        launch_kick(d_particles, N, r->dt);
    }
    cudaDeviceSynchronize();
    sync_to_cpu(r);
}

// ─── Full integrator step ─────────────────────────────────────────────────────

extern "C" void reb_integrator_leapfrog_cuda_step(struct reb_simulation* r) {
    r->gravity_ignore_terms = 0;
    const double dt = r->dt;
    const int N = r->N;

    if (r->ri_leapfrog.order != 2) {
        reb_simulation_error(r, "CUDA leapfrog currently only supports order 2.");
        return;
    }

    sync_to_gpu(r);

    launch_drift(d_particles, N, dt * 0.5);
    cudaDeviceSynchronize();
    sync_to_cpu(r);
    r->t += dt * 0.5;

    reb_simulation_update_acceleration(r);

    sync_to_gpu(r);
    launch_kick(d_particles, N, dt);
    cudaDeviceSynchronize();
    launch_drift(d_particles, N, dt * 0.5);
    cudaDeviceSynchronize();
    sync_to_cpu(r);
    r->t += dt * 0.5;

    r->dt_last_done = dt;
}

extern "C" void reb_integrator_leapfrog_cuda_synchronize(struct reb_simulation* r) {
}

extern "C" void reb_integrator_leapfrog_cuda_reset(struct reb_simulation* r) {
    r->ri_leapfrog.order = 2;
    if (d_particles) {
        cudaFree(d_particles);
        d_particles = NULL;
        d_N_allocated = 0;
    }
}
