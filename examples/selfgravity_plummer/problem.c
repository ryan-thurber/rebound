#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "rebound.h"

/* EDIT: helper to create a simulation with the requested gravity mode */
static struct reb_simulation* create_simulation_with_particles(
    const struct reb_particle* particles,
    int N,
    double G,
    double dt,
    double softening,
    double opening_angle2,
    int gravity_mode,
    double boxsize,
    unsigned int rand_seed
){
    struct reb_simulation* r = reb_simulation_create();
    if (r == NULL){
        return NULL;
    }

    r->rand_seed = rand_seed;
    r->G = G;
    r->gravity = gravity_mode;
    r->integrator = REB_INTEGRATOR_LEAPFROG;
    r->dt = dt;
    r->softening = softening;
    r->opening_angle2 = opening_angle2;

    reb_simulation_configure_box(r, boxsize, 1, 1, 1);
    for (int i = 0; i < N; i++){
        reb_simulation_add(r, particles[i]);
    }
    reb_simulation_move_to_com(r);
    return r;
}

/* EDIT: helper for wall-clock timing */
static double elapsed_seconds(const struct timespec* start, const struct timespec* end){
    return (double)(end->tv_sec - start->tv_sec) + 1.0e-9*(double)(end->tv_nsec - start->tv_nsec);
}

int main(int argc, char* argv[]){
	const int particle_counts[] = {100, 1000, 5000, 10000};
	const int num_counts = (int)(sizeof(particle_counts) / sizeof(particle_counts[0]));
	int nsteps = 100;                 /* EDIT: number of time steps to compare */
    double G = 1.0;
    double M = 1.0;
    double R = 1.0;
    double E = 3.0/64.0*M_PI*M*M/R;
    double r0 = 16.0/(3.0*M_PI)*R;
    double opening_angle2 = 0.25;

	printf("%-8s %-18s %-18s %-14s %-24s\n", "N", "CPU time (s)", "GPU time (s)", "Speed Up", "Avg Pos Error");

	for(int icount = 0; icount < num_counts; icount++){
		int _N = particle_counts[icount];
		double t0 = G*pow(M,5.0/2.0)*pow(4.0*E,-3.0/2.0)*(double)_N/log(0.4*(double)_N);
    	double dt = 2e-5*t0;
    	double softening = 0.01*r0;
		double boxsize = 1000.0*r0;

		struct reb_simulation* r_init = reb_simulation_create();
		if (r_init == NULL){
		    printf("Failed to create initial simulation.\n");
		    return 1;
		}
		
		r_init->rand_seed = 0;
		r_init->G = G;
		r_init->gravity = REB_GRAVITY_TREE;
		r_init->integrator = REB_INTEGRATOR_LEAPFROG;
		r_init->dt = dt;
		r_init->softening = softening;
		r_init->opening_angle2 = opening_angle2;

		reb_simulation_configure_box(r_init, boxsize, 1, 1, 1);
		reb_simulation_add_plummer(r_init, _N, M, R);
		reb_simulation_move_to_com(r_init);
		
		/* EDIT: save identical initial particles for both simulations */
		struct reb_particle* initial_particles = malloc((size_t)r_init->N * sizeof(struct reb_particle));
		if (initial_particles == NULL){
		    printf("Failed to allocate initial particle array.\n");
		    reb_simulation_free(r_init);
		    return 1;
		}
		for (int i = 0; i < r_init->N; i++){
		    initial_particles[i] = r_init->particles[i];
		}

		/* EDIT: create matching CPU and GPU simulations */
		struct reb_simulation* r_cpu = create_simulation_with_particles(
		    initial_particles, r_init->N, G, dt, softening, opening_angle2,
		    REB_GRAVITY_TREE, boxsize, 0
		);
		struct reb_simulation* r_gpu = create_simulation_with_particles(
		    initial_particles, r_init->N, G, dt, softening, opening_angle2,
		    REB_GRAVITY_TREE_GPU, boxsize, 0
		);

		reb_simulation_free(r_init);

		if (r_cpu == NULL || r_gpu == NULL){
		    printf("Failed to create CPU/GPU simulations.\n");
		    free(initial_particles);
		    if (r_cpu != NULL) reb_simulation_free(r_cpu);
		    if (r_gpu != NULL) reb_simulation_free(r_gpu);
		    return 1;
		}

		/* EDIT: run both simulations, time them separately, and accumulate average position error */
		double cpu_time = 0.0;
		double gpu_time = 0.0;
		double error_sum = 0.0;
		long long error_count = 0;

		for (int step = 0; step < nsteps; step++){
		    struct timespec cpu_start, cpu_end, gpu_start, gpu_end;
		    double tnext_cpu = r_cpu->t + r_cpu->dt;
		    double tnext_gpu = r_gpu->t + r_gpu->dt;

		    clock_gettime(CLOCK_MONOTONIC, &cpu_start);
		    reb_simulation_integrate(r_cpu, tnext_cpu);
		    clock_gettime(CLOCK_MONOTONIC, &cpu_end);
		    cpu_time += elapsed_seconds(&cpu_start, &cpu_end);

		    clock_gettime(CLOCK_MONOTONIC, &gpu_start);
		    reb_simulation_integrate(r_gpu, tnext_gpu);
		    clock_gettime(CLOCK_MONOTONIC, &gpu_end);
		    gpu_time += elapsed_seconds(&gpu_start, &gpu_end);

		    for (int i = 0; i < r_cpu->N && i < r_gpu->N; i++){
		        const double dx = r_cpu->particles[i].x - r_gpu->particles[i].x;
		        const double dy = r_cpu->particles[i].y - r_gpu->particles[i].y;
		        const double dz = r_cpu->particles[i].z - r_gpu->particles[i].z;
		        error_sum += sqrt(dx*dx + dy*dy + dz*dz);
		        error_count++;
		    }
		}

		printf("%-8d %-18.9f %-18.9f %-14.6f %-24.12e \n", _N, cpu_time, gpu_time, (gpu_time > 0.0) ? (cpu_time / gpu_time) : 0.0, (error_count > 0) ? error_sum /((double)error_count) : 0.0);

		reb_simulation_free(r_cpu);
		reb_simulation_free(r_gpu);
		free(initial_particles);
	}
    return 0;
}
