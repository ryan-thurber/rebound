#include "gravity_gpu.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef REB_GPU_STACK_MAX
#define REB_GPU_STACK_MAX 128
#endif

#define CUDA_CHECK(call)                                                                              \
	do{                                                                                               \
		cudaError_t err__ = (call);                                                                   \
		if(err__ != cudaSuccess){                                                                     \
			fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
			return 1;                                                                                 \
		}                                                                                             \
	}while(0)

__global__ void reb_bh_force_kernel(int N,
									const double* px,
									const double* py,
									const double* pz,
									const double* pm,
									const double* node_m,
									const double* node_mx,
									const double* node_my,
									const double* node_mz,
									const double* node_w,
									const int* node_pt,
									const int* node_remote,
									const int* child,
									const int* roots,
									int n_roots,
									/* EDIT: current ghost-box shift for this launch */
									double ghost_shift_x,
									double ghost_shift_y,
									double ghost_shift_z,
									double G,
									double softening2,
									double opening_angle2,
									double* ax,
									double* ay,
									double* az){
	
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid >= N) return;

	/* EDIT: shift the source particle into the current ghost box */
	const double x = px[tid] + ghost_shift_x;
	const double y = py[tid] + ghost_shift_y;
	const double z = pz[tid] + ghost_shift_z;

	double ax_local = 0.0;
	double ay_local = 0.0;
	double az_local = 0.0;

	int stack[REB_GPU_STACK_MAX];
	int sp = 0;

	for(int r = 0; r < n_roots; r++){
		int root = roots[r];
		if(root >= 0 && sp < REB_GPU_STACK_MAX){
			stack[sp++] = root;
		}
	}

	while(sp >0){
		int node = stack[--sp];
		
		const double dx = x - node_mx[node];
		const double dy = y - node_my[node];
		const double dz = z - node_mz[node];
		const double r2 = dx*dx + dy*dy + dz*dz;
		
		if(node_pt[node] < 0){
			const double node_width = node_w[node];
			if(node_width * node_width > opening_angle2 * r2){
				for(int o = 0; o < 8; o++){
					int c = child[8*node + o];
					if(c >= 0 && sp < REB_GPU_STACK_MAX){
						stack[sp++] = c;
					}
				}
			} else{
				double rr = sqrt(r2 + softening2);
				double prefact = -G * node_m[node] / (rr * rr * rr);

				ax_local += prefact * dx;
				ay_local += prefact * dy;
				az_local += prefact * dz;
			}
		} else {
			/* EDIT: only suppress self-interaction for the central box.
			   Ghost-box images should still contribute. */
			if(!(ghost_shift_x == 0.0 && ghost_shift_y == 0.0 && ghost_shift_z == 0.0 &&
			     node_remote[node] == 0 && node_pt[node] == tid)){
				double rr = sqrt(r2 + softening2);
				double prefact = -G * node_m[node] / (rr * rr * rr);

				ax_local += prefact * dx;
				ay_local += prefact * dy;
				az_local += prefact * dz;
			}
		}
	}
	
	/* EDIT: accumulate because the wrapper launches once per ghost box */
	ax[tid] += ax_local;
	ay[tid] += ay_local;
	az[tid] += az_local;
}

extern "C" int reb_gpu_bh_accel_raw(int N,
						 const double* px,
						 const double* py,
						 const double* pz,
						 const double* pm,
						 const double* node_m,
						 const double* node_mx,
						 const double* node_my,
						 const double* node_mz,
						 const double* node_w,
						 const int* node_pt,
						 const int* node_remote,
						 const int* child,
						 const int* roots,
						 int n_nodes,
						 int n_roots,
						 /* EDIT: ghost-box shifts passed in from bridge */
						 const double* ghost_shift_x,
						 const double* ghost_shift_y,
						 const double* ghost_shift_z,
						 int n_ghost_boxes,
						 double G,
						 double softening2,
						 double opening_angle2,
						 double* ax,
						 double* ay,
						 double* az){
	
	if(N <= 0 || n_roots < 0 || n_ghost_boxes <= 0 || !px || !py || !pz || !node_m || !node_mx || !node_my || !node_mz || !node_w || !node_pt || !node_remote || !child || !roots || !ghost_shift_x || !ghost_shift_y || !ghost_shift_z || !ax || !ay || !az){
		fprintf(stderr, "reb_gpu_bh_accel_raw: invalid input!");
		return 0;
	}

	double *d_px = NULL;
	double *d_py = NULL;
	double *d_pz = NULL;
	double *d_node_m = NULL;
	double *d_node_mx = NULL;
	double *d_node_my = NULL;
	double *d_node_mz = NULL;
	double *d_node_w = NULL;
	int *d_node_pt = NULL;
	int *d_node_remote = NULL;
	int *d_child = NULL;
	int *d_roots = NULL;
	double *d_ax = NULL;
	double *d_ay = NULL;
	double *d_az = NULL;

	const size_t pbytes = (size_t)N * sizeof(double);
	const size_t nbytes_d = (size_t)n_nodes * sizeof(double);
	const size_t nbytes_i = (size_t)n_nodes * sizeof(int);
	const size_t cbytes = (size_t)(8 * n_nodes) * sizeof(int);
	const size_t rbytes = (size_t)n_roots * sizeof(int);

	CUDA_CHECK(cudaMalloc((void**)&d_px, pbytes));
	CUDA_CHECK(cudaMalloc((void**)&d_py, pbytes));
	CUDA_CHECK(cudaMalloc((void**)&d_pz, pbytes));

	CUDA_CHECK(cudaMalloc((void**)&d_node_m, nbytes_d));
	CUDA_CHECK(cudaMalloc((void**)&d_node_mx, nbytes_d));
	CUDA_CHECK(cudaMalloc((void**)&d_node_my, nbytes_d));
	CUDA_CHECK(cudaMalloc((void**)&d_node_mz, nbytes_d));
	CUDA_CHECK(cudaMalloc((void**)&d_node_w, nbytes_d));

	CUDA_CHECK(cudaMalloc((void**)&d_node_pt, nbytes_i));
	CUDA_CHECK(cudaMalloc((void**)&d_node_remote, nbytes_i));
	CUDA_CHECK(cudaMalloc((void**)&d_child, cbytes));
	CUDA_CHECK(cudaMalloc((void**)&d_roots, rbytes));
	
	CUDA_CHECK(cudaMalloc((void**)&d_ax, pbytes));
	CUDA_CHECK(cudaMalloc((void**)&d_ay, pbytes));
	CUDA_CHECK(cudaMalloc((void**)&d_az, pbytes));

	CUDA_CHECK(cudaMemcpy(d_px, px, pbytes, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_py, py, pbytes, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_pz, pz, pbytes, cudaMemcpyHostToDevice));

	CUDA_CHECK(cudaMemcpy(d_node_m, node_m, nbytes_d, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_node_mx, node_mx, nbytes_d, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_node_my, node_my, nbytes_d, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_node_mz, node_mz, nbytes_d, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_node_w, node_w, nbytes_d, cudaMemcpyHostToDevice));

	CUDA_CHECK(cudaMemcpy(d_node_pt, node_pt, nbytes_i, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_node_remote, node_remote, nbytes_i, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_child, child, cbytes, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_roots, roots, rbytes, cudaMemcpyHostToDevice));

	CUDA_CHECK(cudaMemcpy(d_ax, ax, pbytes, cudaMemcpyHostToDevice)); /* EDIT: keep host-side zero init */
	CUDA_CHECK(cudaMemcpy(d_ay, ay, pbytes, cudaMemcpyHostToDevice)); /* EDIT: keep host-side zero init */
	CUDA_CHECK(cudaMemcpy(d_az, az, pbytes, cudaMemcpyHostToDevice)); /* EDIT: keep host-side zero init */
	
	dim3 blockDim(128);
	dim3 gridDim((N + 128 - 1)/ 128);

	cudaError_t err = cudaSuccess;

	for(int g = 0; g < n_ghost_boxes; g++){
		reb_bh_force_kernel<<<gridDim, blockDim>>>(N,
									 d_px,
									 d_py,
									 d_pz,
									 NULL,
									 d_node_m,
									 d_node_mx,
									 d_node_my,
									 d_node_mz,
									 d_node_w,
									 d_node_pt,
									 d_node_remote,
									 d_child,
									 d_roots,
									 n_roots,
									 ghost_shift_x[g],
									 ghost_shift_y[g],
									 ghost_shift_z[g],
									 G,
									 softening2,
									 opening_angle2,
									 d_ax,
									 d_ay,
									 d_az);

		err = cudaGetLastError();
		if(err != cudaSuccess){
			fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
			goto fail;
		}
	}
	
	err = cudaDeviceSynchronize();
	if(err != cudaSuccess){
		fprintf(stderr, "Kernel execution failed: %s\n", cudaGetErrorString(err));
		goto fail;
	}

	CUDA_CHECK(cudaMemcpy(ax, d_ax, pbytes, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(ay, d_ay, pbytes, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(az, d_az, pbytes, cudaMemcpyDeviceToHost));
	
	cudaFree(d_px);
	cudaFree(d_py);
	cudaFree(d_pz);
	cudaFree(d_node_m);
	cudaFree(d_node_mx);
	cudaFree(d_node_my);
	cudaFree(d_node_mz);
	cudaFree(d_node_w);
	cudaFree(d_node_pt);
	cudaFree(d_node_remote);
	cudaFree(d_child);
	cudaFree(d_roots);
	cudaFree(d_ax);
	cudaFree(d_ay);
	cudaFree(d_az);
	return 0;
	
	fail:
		cudaFree(d_px);
		cudaFree(d_py);
		cudaFree(d_pz);
		cudaFree(d_node_m);
		cudaFree(d_node_mx);
		cudaFree(d_node_my);
		cudaFree(d_node_mz);
		cudaFree(d_node_w);
		cudaFree(d_node_pt);
		cudaFree(d_node_remote);
		cudaFree(d_child);
		cudaFree(d_roots);
		cudaFree(d_ax);
		cudaFree(d_ay);
		cudaFree(d_az);
		return 1;
}
