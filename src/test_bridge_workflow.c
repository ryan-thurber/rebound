#include <stdio.h>
#include <math.h>

#include "rebound.h"
#include "tree.h"
#include "gravity_gpu_bridge.h"

static int nearly_equal(double a, double b, double tol){
    return fabs(a - b) < tol;
}

int main(void){
    const double tol = 1e-10;
    int pass = 1;
    int i;

    struct reb_simulation* r = reb_simulation_create();
    if (r == NULL){
        printf("Failed to create simulation.\n");
        return 1;
    }

    /* Use tree gravity */
    r->gravity = REB_GRAVITY_TREE;
    r->G = 1.0;
    r->softening = 0.0;
    r->opening_angle2 = 0.25;

	reb_simulation_configure_box(r, 10.0, 1, 1, 1);

    /* Two-particle test */
    struct reb_particle p0 = {0};
    p0.x = 0.0;
    p0.y = 0.0;
    p0.z = 0.0;
    p0.m = 1.0;

    struct reb_particle p1 = {0};
    p1.x = 1.0;
    p1.y = 0.0;
    p1.z = 0.0;
    p1.m = 1.0;

    reb_simulation_add(r, p0);
    reb_simulation_add(r, p1);

    /* Zero accelerations before calling bridge */
    for (i = 0; i < r->N; i++){
        r->particles[i].ax = 0.0;
        r->particles[i].ay = 0.0;
        r->particles[i].az = 0.0;
    }

    /* Call your bridge: REBOUND struct -> flat arrays -> CUDA -> REBOUND struct */
    if (reb_gpu_tree_accel_bridge(r) != 0){
        printf("reb_gpu_tree_accel_bridge failed.\n");
        reb_simulation_free(r);
        return 1;
    }

    printf("Returned accelerations:\n");
    for (i = 0; i < r->N; i++){
        printf("particle %d: ax = %.12f, ay = %.12f, az = %.12f\n",
               i, r->particles[i].ax, r->particles[i].ay, r->particles[i].az);
    }

    /* Expected result */
    if (!nearly_equal(r->particles[0].ax,  1.0, tol)) pass = 0;
    if (!nearly_equal(r->particles[0].ay,  0.0, tol)) pass = 0;
    if (!nearly_equal(r->particles[0].az,  0.0, tol)) pass = 0;

    if (!nearly_equal(r->particles[1].ax, -1.0, tol)) pass = 0;
    if (!nearly_equal(r->particles[1].ay,  0.0, tol)) pass = 0;
    if (!nearly_equal(r->particles[1].az,  0.0, tol)) pass = 0;

    if (pass){
        printf("BRIDGE WORKFLOW TEST PASSED\n");
    } else {
        printf("BRIDGE WORKFLOW TEST FAILED\n");
    }

    reb_simulation_free(r);
    return pass ? 0 : 1;
}
