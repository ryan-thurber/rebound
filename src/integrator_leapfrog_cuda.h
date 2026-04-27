#ifndef _INTEGRATOR_LEAPFROG_CUDA_H
#define _INTEGRATOR_LEAPFROG_CUDA_H

#include "rebound.h"

void reb_integrator_leapfrog_cuda_step(struct reb_simulation* r);
void reb_integrator_leapfrog_cuda_synchronize(struct reb_simulation* r);
void reb_integrator_leapfrog_cuda_reset(struct reb_simulation* r);

#endif