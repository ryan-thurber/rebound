#include "gravity_opt.h"

// Pulled from hw4
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
            file, line);
    if (abort)
      exit(code);
  }
}

// Naive kernel - basically a direct port of host code to run on GPU
__global__ void gravity_basic_naive(int N_real, int N_active, double G, double softening2, unsigned int gravity_ignore_terms,
                                    double gbx, double gby, double gbz,
                                    double *x, double *y, double *z, const double *m,
                                    double *ax, double *ay, double *az){
    // N_real, N_active are loop bounds
    // G, softening2, and gravity_ignore_terms are constants (could be moved to const memory?)
    // gbx, gby, gbz arrays store the ghost box x, y, and z
    // x, y, and z arrays store the particle x, y, and z
    // m array stores the particle mass
    // ax, ay, az are the output varaiables for acceleration computed for the particle
    // Note: particle 0 corresponds to x[0], y[0] , z[0], m[0], ax[0], ay[0], az[0]
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N_real){
        // Set x,y,z postiton relative to ghost box
        double xi = x[i] + gbx;
        double yi = y[i] + gby;
        double zi = z[i] + gbz;
        // Initialize registers to be used in the calculation
        double dx, dy, dz, _r, prefact;
        double _ax = 0.0, _ay = 0.0, _az = 0.0;
        for (int j = 0; j < N_active; j++){
            if (gravity_ignore_terms==1 && ((j==1 && i==0) || (i==1 && j==0) )) continue;
            if (gravity_ignore_terms==2 && ((j==0 || i==0) )) continue;
            if (i==j) continue;
            dx = xi - x[j];
            dy = yi - y[j];
            dz = zi - z[j];
            _r = sqrt(dx*dx + dy*dy + dz*dz + softening2);
            prefact = -G/(_r*_r*_r)*m[j];

            _ax    += prefact*dx;
            _ay    += prefact*dy;
            _az    += prefact*dz;
        }
        ax[i] = _ax;
        ay[i] = _ay;
        az[i] = _az;
    }
}

// Need a wrapper to launch the kernel from C code
extern "C" void launch_gravity_basic_naive(int N_real, int N_active, double G, double softening2, unsigned int gravity_ignore_terms,
                                    struct reb_vec6d *gb, struct reb_particle *particles) {
    // Host arrays
    double *host_x;
    double *host_y;
    double *host_z;
    double *host_m;
    double *host_ax;
    double *host_ay;
    double *host_az;
    // Device arrays
    double *device_x;
    double *device_y;
    double *device_z;
    double *device_m;
    double *device_ax;
    double *device_ay;
    double *device_az;

    int max_N = (N_real > N_active) ? N_real : N_active;

    // Allocate host memory
    host_x = (double*)malloc(sizeof(double)*max_N);
    host_y = (double*)malloc(sizeof(double)*max_N);
    host_z = (double*)malloc(sizeof(double)*max_N);
    host_m = (double*)malloc(sizeof(double)*max_N);
    host_ax = (double*)malloc(sizeof(double)*N_real);
    host_ay = (double*)malloc(sizeof(double)*N_real);
    host_az = (double*)malloc(sizeof(double)*N_real);

    // Initialize host arrays from particles
    for (int i=0;i<max_N;i++){
        host_x[i] = particles[i].x;
        host_y[i] = particles[i].y;
        host_z[i] = particles[i].z;
        host_m[i] = particles[i].m;
    }

    // Allocate device memory

    CUDA_CHECK(cudaMalloc((void **) &device_x, sizeof(double)*max_N));
    CUDA_CHECK(cudaMalloc((void **) &device_y, sizeof(double)*max_N));
    CUDA_CHECK(cudaMalloc((void **) &device_z, sizeof(double)*max_N));
    CUDA_CHECK(cudaMalloc((void **) &device_m, sizeof(double)*max_N));
    CUDA_CHECK(cudaMalloc((void **) &device_ax, sizeof(double)*N_real));
    CUDA_CHECK(cudaMalloc((void **) &device_ay, sizeof(double)*N_real));
    CUDA_CHECK(cudaMalloc((void **) &device_az, sizeof(double)*N_real));

    // Transfer host arrays to device
    CUDA_CHECK(cudaMemcpy(device_x, host_x, sizeof(double)*max_N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_y, host_y, sizeof(double)*max_N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_z, host_z, sizeof(double)*max_N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_m, host_m, sizeof(double)*max_N, cudaMemcpyHostToDevice));

    int threads = 128;
    int blocks = (N_real + threads - 1) / threads;

    //Launch kernel
    gravity_basic_naive<<<blocks,threads>>>(N_real, N_active, G, softening2, gravity_ignore_terms,
                                gb->x, gb->y, gb->z,
                                device_x, device_y, device_z, device_m,
                                device_ax, device_ay, device_az);
    cudaDeviceSynchronize();

    // Transfer device arrays to host
    CUDA_CHECK(cudaMemcpy(host_ax, device_ax, sizeof(double)*N_real, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_ay, device_ay, sizeof(double)*N_real, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_az, device_az, sizeof(double)*N_real, cudaMemcpyDeviceToHost));

    // Set particle result accelerations
    for (int i=0;i<N_real;i++){
        particles[i].ax = host_ax[i];
        particles[i].ay = host_ay[i];
        particles[i].az = host_az[i];
    }

    // Free host and device memory
    free(host_x);
    free(host_y);
    free(host_z);
    free(host_m);
    free(host_ax);
    free(host_ay);
    free(host_az);
    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_z);
    cudaFree(device_m);
    cudaFree(device_ax);
    cudaFree(device_ay);
    cudaFree(device_az);

}

__constant__ double d_G;
__constant__ double d_softening2;
__constant__ unsigned int d_gravity_ignore_terms;

// Optimization 1 - Shared memory and const memory
__global__ void gravity_basic_opt_1(int N_real, int N_active,
                                    double gbx, double gby, double gbz,
                                    double *x, double *y, double *z, const double *m,
                                    double *ax, double *ay, double *az){
    // N_real, N_active are loop bounds
    // G, softening2, and gravity_ignore_terms are constants (could be moved to const memory?)
    // gbx, gby, gbz arrays store the ghost box x, y, and z
    // x, y, and z arrays store the particle x, y, and z
    // m array stores the particle mass
    // ax, ay, az are the output varaiables for acceleration computed for the particle
    // Note: particle 0 corresponds to x[0], y[0] , z[0], m[0], ax[0], ay[0], az[0]
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N_real){
        // Set x,y,z postiton relative to ghost box
        double xi = x[i] + gbx;
        double yi = y[i] + gby;
        double zi = z[i] + gbz;
        // Initialize registers to be used in the calculation
        double dx, dy, dz, _r, prefact;
        double _ax = 0.0, _ay = 0.0, _az = 0.0;
        for (int j = 0; j < N_active; j++){
            if (d_gravity_ignore_terms==1 && ((j==1 && i==0) || (i==1 && j==0) )) continue;
            if (d_gravity_ignore_terms==2 && ((j==0 || i==0) )) continue;
            if (i==j) continue;
            dx = xi - x[j];
            dy = yi - y[j];
            dz = zi - z[j];
            _r = sqrt(dx*dx + dy*dy + dz*dz + d_softening2);
            prefact = -d_G/(_r*_r*_r)*m[j];

            _ax    += prefact*dx;
            _ay    += prefact*dy;
            _az    += prefact*dz;
        }
        ax[i] = _ax;
        ay[i] = _ay;
        az[i] = _az;
    }
}

// Need a wrapper to launch the kernel from C code
extern "C" void launch_gravity_basic_opt_1(int N_real, int N_active, double G, double softening2, unsigned int gravity_ignore_terms,
                                    struct reb_vec6d *gb, struct reb_particle *particles) {
    // Host arrays
    double *host_x;
    double *host_y;
    double *host_z;
    double *host_m;
    double *host_ax;
    double *host_ay;
    double *host_az;
    // Device arrays
    double *device_x;
    double *device_y;
    double *device_z;
    double *device_m;
    double *device_ax;
    double *device_ay;
    double *device_az;

    int max_N = (N_real > N_active) ? N_real : N_active;

    // Allocate host memory
    host_x = (double*)malloc(sizeof(double)*max_N);
    host_y = (double*)malloc(sizeof(double)*max_N);
    host_z = (double*)malloc(sizeof(double)*max_N);
    host_m = (double*)malloc(sizeof(double)*max_N);
    host_ax = (double*)malloc(sizeof(double)*N_real);
    host_ay = (double*)malloc(sizeof(double)*N_real);
    host_az = (double*)malloc(sizeof(double)*N_real);

    // Initialize host arrays from particles
    for (int i=0;i<max_N;i++){
        host_x[i] = particles[i].x;
        host_y[i] = particles[i].y;
        host_z[i] = particles[i].z;
        host_m[i] = particles[i].m;
    }

    // Allocate device memory

    CUDA_CHECK(cudaMalloc((void **) &device_x, sizeof(double)*max_N));
    CUDA_CHECK(cudaMalloc((void **) &device_y, sizeof(double)*max_N));
    CUDA_CHECK(cudaMalloc((void **) &device_z, sizeof(double)*max_N));
    CUDA_CHECK(cudaMalloc((void **) &device_m, sizeof(double)*max_N));
    CUDA_CHECK(cudaMalloc((void **) &device_ax, sizeof(double)*N_real));
    CUDA_CHECK(cudaMalloc((void **) &device_ay, sizeof(double)*N_real));
    CUDA_CHECK(cudaMalloc((void **) &device_az, sizeof(double)*N_real));

    // Transfer host arrays to device
    CUDA_CHECK(cudaMemcpy(device_x, host_x, sizeof(double)*max_N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_y, host_y, sizeof(double)*max_N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_z, host_z, sizeof(double)*max_N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_m, host_m, sizeof(double)*max_N, cudaMemcpyHostToDevice));

    // Transfer constants
    CUDA_CHECK(cudaMemcpyToSymbol(G, &d_G, sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(softening2, &d_softening2, sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(gravity_ignore_terms, &d_gravity_ignore_terms, sizeof(unsigned int)));

    int threads = 128;
    int blocks = (N_real + threads - 1) / threads;

    //Launch kernel
    gravity_basic_opt_1<<<blocks,threads>>>(N_real, N_active,
                                gb->x, gb->y, gb->z,
                                device_x, device_y, device_z, device_m,
                                device_ax, device_ay, device_az);
    cudaDeviceSynchronize();

    // Transfer device arrays to host
    CUDA_CHECK(cudaMemcpy(host_ax, device_ax, sizeof(double)*N_real, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_ay, device_ay, sizeof(double)*N_real, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_az, device_az, sizeof(double)*N_real, cudaMemcpyDeviceToHost));

    // Set particle result accelerations
    for (int i=0;i<N_real;i++){
        particles[i].ax = host_ax[i];
        particles[i].ay = host_ay[i];
        particles[i].az = host_az[i];
    }

    // Free host and device memory
    free(host_x);
    free(host_y);
    free(host_z);
    free(host_m);
    free(host_ax);
    free(host_ay);
    free(host_az);
    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_z);
    cudaFree(device_m);
    cudaFree(device_ax);
    cudaFree(device_ay);
    cudaFree(device_az);

}