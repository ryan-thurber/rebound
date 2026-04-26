#include "rebound.h"
#include "tree.h"
#include "gravity_gpu.h"
#include "boundary.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Public entry point to call from gravity.c */
int reb_gpu_tree_accel_bridge(struct reb_simulation* r);

/* Flat tree container used only on the host side */
struct reb_gpu_flat_tree {
    int n_nodes;
    int n_roots;

    double* node_m;
    double* node_mx;
    double* node_my;
    double* node_mz;
    double* node_w;

    int* node_pt;
    int* node_remote;

    int* child;   /* length = 8*n_nodes */
    int* roots;   /* length = n_roots */
};

struct reb_gpu_flatten_ctx {
    struct reb_gpu_flat_tree* ft;
    int next_id;
};

/* EDIT: flat list of ghost-box position shifts to mirror CPU tree gravity loops */
struct reb_gpu_ghost_shifts {
    int n_ghost_boxes;
    double* x;
    double* y;
    double* z;
};

static void reb_gpu_free_flat_tree(struct reb_gpu_flat_tree* ft){
    if (!ft) return;

    free(ft->node_m);
    free(ft->node_mx);
    free(ft->node_my);
    free(ft->node_mz);
    free(ft->node_w);

    free(ft->node_pt);
    free(ft->node_remote);

    free(ft->child);
    free(ft->roots);

    memset(ft, 0, sizeof(*ft));
}

static void reb_gpu_free_ghost_shifts(struct reb_gpu_ghost_shifts* gs){
    if (!gs) return;

    free(gs->x);
    free(gs->y);
    free(gs->z);

    memset(gs, 0, sizeof(*gs));
}

static void reb_gpu_free_particle_arrays(
    double* px, double* py, double* pz, double* pm,
    double* ax, double* ay, double* az
){
    free(px);
    free(py);
    free(pz);
    free(pm);
    free(ax);
    free(ay);
    free(az);
}

static int reb_gpu_count_nodes_recursive(const struct reb_treecell* node){
    int count = 1;
    int o;

    if (node == NULL){
        return 0;
    }

    for (o = 0; o < 8; o++){
        count += reb_gpu_count_nodes_recursive(node->oct[o]);
    }

    return count;
}

static int reb_gpu_fill_nodes_recursive(
    const struct reb_treecell* node,
    struct reb_gpu_flatten_ctx* ctx
){
    int my_id;
    int o;

    if (node == NULL){
        return -1;
    }

    my_id = ctx->next_id++;
    ctx->ft->node_m[my_id]      = node->m;
    ctx->ft->node_mx[my_id]     = node->mx;
    ctx->ft->node_my[my_id]     = node->my;
    ctx->ft->node_mz[my_id]     = node->mz;
    ctx->ft->node_w[my_id]      = node->w;
    ctx->ft->node_pt[my_id]     = node->pt;
    ctx->ft->node_remote[my_id] = node->remote;

    for (o = 0; o < 8; o++){
        ctx->ft->child[8*my_id + o] = -1;
    }

    for (o = 0; o < 8; o++){
        if (node->oct[o] != NULL){
            int cid = reb_gpu_fill_nodes_recursive(node->oct[o], ctx);
            ctx->ft->child[8*my_id + o] = cid;
        }
    }

    return my_id;
}

static int reb_gpu_flatten_tree_from_rebound(
    const struct reb_simulation* r,
    struct reb_gpu_flat_tree* ft
){
    int i;
    int total_nodes = 0;
    int root_count = 0;
    struct reb_gpu_flatten_ctx ctx;

    memset(ft, 0, sizeof(*ft));

    if (r == NULL || r->tree_root == NULL){
        fprintf(stderr, "reb_gpu_flatten_tree_from_rebound: tree_root is NULL\n");
        return 1;
    }

    for (i = 0; i < r->N_root; i++){
        if (r->tree_root[i] != NULL){
            total_nodes += reb_gpu_count_nodes_recursive(r->tree_root[i]);
            root_count++;
        }
    }

    if (total_nodes <= 0 || root_count <= 0){
        fprintf(stderr, "reb_gpu_flatten_tree_from_rebound: no tree nodes found\n");
        return 1;
    }

    ft->n_nodes = total_nodes;
    ft->n_roots = root_count;

    ft->node_m  = (double*)malloc((size_t)ft->n_nodes * sizeof(double));
    ft->node_mx = (double*)malloc((size_t)ft->n_nodes * sizeof(double));
    ft->node_my = (double*)malloc((size_t)ft->n_nodes * sizeof(double));
    ft->node_mz = (double*)malloc((size_t)ft->n_nodes * sizeof(double));
    ft->node_w  = (double*)malloc((size_t)ft->n_nodes * sizeof(double));

    ft->node_pt     = (int*)malloc((size_t)ft->n_nodes * sizeof(int));
    ft->node_remote = (int*)malloc((size_t)ft->n_nodes * sizeof(int));

    ft->child = (int*)malloc((size_t)(8 * ft->n_nodes) * sizeof(int));
    ft->roots = (int*)malloc((size_t)ft->n_roots * sizeof(int));

    if (!ft->node_m || !ft->node_mx || !ft->node_my || !ft->node_mz || !ft->node_w ||
        !ft->node_pt || !ft->node_remote || !ft->child || !ft->roots){
        fprintf(stderr, "reb_gpu_flatten_tree_from_rebound: malloc failed\n");
        reb_gpu_free_flat_tree(ft);
        return 1;
    }

    ctx.ft = ft;
    ctx.next_id = 0;

    {
        int rslot = 0;
        for (i = 0; i < r->N_root; i++){
            if (r->tree_root[i] != NULL){
                ft->roots[rslot++] = reb_gpu_fill_nodes_recursive(r->tree_root[i], &ctx);
            }
        }
    }

    if (ctx.next_id != ft->n_nodes){
        fprintf(stderr, "reb_gpu_flatten_tree_from_rebound: node count mismatch\n");
        reb_gpu_free_flat_tree(ft);
        return 1;
    }

    return 0;
}


static int reb_gpu_build_ghost_shifts(
    const struct reb_simulation* r,
    struct reb_gpu_ghost_shifts* gs
){
    int gbx, gby, gbz;
    int idx = 0;

    memset(gs, 0, sizeof(*gs));

    gs->n_ghost_boxes = (2*r->N_ghost_x + 1) *
                        (2*r->N_ghost_y + 1) *
                        (2*r->N_ghost_z + 1);

    gs->x = (double*)malloc((size_t)gs->n_ghost_boxes * sizeof(double));
    gs->y = (double*)malloc((size_t)gs->n_ghost_boxes * sizeof(double));
    gs->z = (double*)malloc((size_t)gs->n_ghost_boxes * sizeof(double));

    if (!gs->x || !gs->y || !gs->z){
        reb_gpu_free_ghost_shifts(gs);
        return 1;
    }

    for (gbx = -r->N_ghost_x; gbx <= r->N_ghost_x; gbx++){
        for (gby = -r->N_ghost_y; gby <= r->N_ghost_y; gby++){
            for (gbz = -r->N_ghost_z; gbz <= r->N_ghost_z; gbz++){
                struct reb_vec6d gb = reb_boundary_get_ghostbox(r, gbx, gby, gbz);
                gs->x[idx] = gb.x;
                gs->y[idx] = gb.y;
                gs->z[idx] = gb.z;
                idx++;
            }
        }
    }

    return 0;
}

static int reb_gpu_extract_particles(
    const struct reb_simulation* r,
    double** px_out,
    double** py_out,
    double** pz_out,
    double** pm_out,
    double** ax_out,
    double** ay_out,
    double** az_out
){
    int i;
    int N = r->N;

    double* px = (double*)malloc((size_t)N * sizeof(double));
    double* py = (double*)malloc((size_t)N * sizeof(double));
    double* pz = (double*)malloc((size_t)N * sizeof(double));
    double* pm = (double*)malloc((size_t)N * sizeof(double));
    double* ax = (double*)malloc((size_t)N * sizeof(double));
    double* ay = (double*)malloc((size_t)N * sizeof(double));
    double* az = (double*)malloc((size_t)N * sizeof(double));

    if (!px || !py || !pz || !pm || !ax || !ay || !az){
        reb_gpu_free_particle_arrays(px, py, pz, pm, ax, ay, az);
        return 1;
    }

    for (i = 0; i < N; i++){
        px[i] = r->particles[i].x;
        py[i] = r->particles[i].y;
        pz[i] = r->particles[i].z;
        pm[i] = r->particles[i].m;

        ax[i] = 0.0;
        ay[i] = 0.0;
        az[i] = 0.0;
    }

    *px_out = px;
    *py_out = py;
    *pz_out = pz;
    *pm_out = pm;
    *ax_out = ax;
    *ay_out = ay;
    *az_out = az;

    return 0;
}

int reb_gpu_tree_accel_bridge(struct reb_simulation* r){
    struct reb_gpu_flat_tree ft;
    double *px = NULL, *py = NULL, *pz = NULL, *pm = NULL;
    double *ax = NULL, *ay = NULL, *az = NULL;
    struct reb_gpu_ghost_shifts gs; /* EDIT */
    int i;
    int status;

    memset(&ft, 0, sizeof(ft));
    memset(&gs, 0, sizeof(gs)); /* EDIT */

    if (r == NULL){
        fprintf(stderr, "reb_gpu_tree_accel_bridge: r is NULL\n");
        return 1;
    }

    if (r->N <= 0){
        fprintf(stderr, "reb_gpu_tree_accel_bridge: no particles\n");
        return 1;
    }

    /* Make sure the CPU tree exists and its COM/mass data are current. */
    if (r->tree_needs_update){
        reb_simulation_update_tree(r);
    }
    reb_simulation_update_tree_gravity_data(r);

    if (reb_gpu_extract_particles(r, &px, &py, &pz, &pm, &ax, &ay, &az) != 0){
        fprintf(stderr, "reb_gpu_tree_accel_bridge: particle extraction failed\n");
        return 1;
    }

    if (reb_gpu_flatten_tree_from_rebound(r, &ft) != 0){
        fprintf(stderr, "reb_gpu_tree_accel_bridge: tree flatten failed\n");
        reb_gpu_free_particle_arrays(px, py, pz, pm, ax, ay, az);
        return 1;
    }

    if (reb_gpu_build_ghost_shifts(r, &gs) != 0){ /* EDIT */
        fprintf(stderr, "reb_gpu_tree_accel_bridge: ghost shift build failed\n");
        reb_gpu_free_flat_tree(&ft);
        reb_gpu_free_ghost_shifts(&gs); /* EDIT */
        reb_gpu_free_particle_arrays(px, py, pz, pm, ax, ay, az);
        return 1;
    }

    status = reb_gpu_bh_accel_raw(
        r->N,
        px, py, pz, pm,
        ft.node_m,
        ft.node_mx,
        ft.node_my,
        ft.node_mz,
        ft.node_w,
        ft.node_pt,
        ft.node_remote,
        ft.child,
        ft.roots,
        ft.n_nodes,
        ft.n_roots,
        /* EDIT: pass ghost-box shifts down to CUDA side */
        gs.x,
        gs.y,
        gs.z,
        gs.n_ghost_boxes,
        r->G,
        r->softening * r->softening,
        r->opening_angle2,
        ax, ay, az
    );

    if (status != 0){
        fprintf(stderr, "reb_gpu_tree_accel_bridge: CUDA call failed\n");
        reb_gpu_free_flat_tree(&ft);
        reb_gpu_free_particle_arrays(px, py, pz, pm, ax, ay, az);
        return 1;
    }

    for (i = 0; i < r->N; i++){
        r->particles[i].ax = ax[i];
        r->particles[i].ay = ay[i];
        r->particles[i].az = az[i];
    }

    reb_gpu_free_flat_tree(&ft);
    reb_gpu_free_ghost_shifts(&gs); /* EDIT */
    reb_gpu_free_particle_arrays(px, py, pz, pm, ax, ay, az);

    return 0;
}
