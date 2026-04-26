#ifndef REB_GRAVITY_GPU_H /*Ensures this header is only imported once*/
#define REB_GRAVITY_GPU_H

#ifdef __cplusplus /*Ensures that this complies properly in a C++ and C enviroment*/
extern "C" {
#endif

int reb_gpu_bh_accel_raw(int N,
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
                         double* az);

#ifdef __cplusplus
}
#endif

#endif
