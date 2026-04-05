#ifndef REBOUND_SRC_GRAVITY_OPT_H
#define REBOUND_SRC_GRAVITY_OPT_H

#ifdef __cplusplus
extern "C" {
#endif

#include "rebound.h"

void launch_gravity_basic_naive(int N_real, int N_active, double G, double softening2, unsigned int gravity_ignore_terms,
                                    reb_vec6d *gb, reb_particle *particles);

#ifdef __cplusplus
}
#endif

#endif // REBOUND_SRC_GRAVITY_OPT_H