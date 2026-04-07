/**
 * @file integrator_leapfrog_cuda.cu
 * @brief CUDA implementation of the Leap-frog integration scheme.
 */

#include <cuda_runtime.h>
#include <stdio.h>

extern "C" {
#include "rebound.h"
#include "integrator_leapfrog_cuda.h"
}

#define BLOCK_SIZE 256


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


static struct reb_particle* d_particles = NULL;
static int d_N_allocated = 0;

static void sync_to_gpu(struct reb_simulation* r) {
    int N = r->N;
    if (N > d_N_allocated) {
        if (d_particles) cudaFree(d_particles);
        cudaMalloc(&d_particles, N * sizeof(struct reb_particle));
        d_N_allocated = N;
    }
    cudaMemcpy(d_particles, r->particles, N * sizeof(struct reb_particle), cudaMemcpyHostToDevice);
}

static void sync_to_cpu(struct reb_simulation* r) {
    cudaMemcpy(r->particles, d_particles, r->N * sizeof(struct reb_particle), cudaMemcpyDeviceToHost);
}


static void launch_drift(int N, double dt) {
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    drift_kernel<<<blocks, BLOCK_SIZE>>>(d_particles, N, dt);
}

static void launch_kick(int N, double dt) {
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kick_kernel<<<blocks, BLOCK_SIZE>>>(d_particles, N, dt);
}


extern "C" void reb_integrator_leapfrog_cuda_step(struct reb_simulation* r) {
    r->gravity_ignore_terms = 0;
    const double dt = r->dt;
    const int N = r->N;

    // Currently only supporting standard 2nd order via CUDA
    if (r->ri_leapfrog.order != 2) {
        reb_simulation_error(r, "CUDA leapfrog currently only supports order 2.");
        return;
    }

    // 1. Initial Half-Drift (x = x + v * dt/2)
    sync_to_gpu(r);
    launch_drift(N, dt * 0.5);
    cudaDeviceSynchronize();
    
    // 2. Move data back to CPU to update accelerations based on new positions
    sync_to_cpu(r);
    r->t += dt * 0.5;

    // 3. Update Accelerations (CPU Gravity Solver)
    // This fills particles[i].ax, .ay, .az
    reb_simulation_update_acceleration(r);

    // 4. Kick (v = v + a * dt)
    // Push the new accelerations back to the GPU
    sync_to_gpu(r);
    launch_kick(N, dt);
    cudaDeviceSynchronize();

    // 5. Final Half-Drift (x = x + v * dt/2)
    launch_drift(N, dt * 0.5);
    cudaDeviceSynchronize();

    // 6. Final Sync to CPU for output/next step
    sync_to_cpu(r);
    r->t += dt * 0.5;

    r->dt_last_done = dt;
}


extern "C" void reb_integrator_leapfrog_cuda_synchronize(struct reb_simulation* r) {
    // Used if external forces need to be synced; currently handled within step.
}

extern "C" void reb_integrator_leapfrog_cuda_reset(struct reb_simulation* r) {
    r->ri_leapfrog.order = 2;
    if (d_particles) {
        cudaFree(d_particles);
        d_particles = NULL;
        d_N_allocated = 0;
    }
}
