#ifndef MOE_H
#define MOE_H

#include "model.h"

void moe(Model *m, Layer *l, int layer, float *x, int S, float *out);
void dense_mlp(Layer *l, float *x, int S, int D, int I, float *out);

#endif /* MOE_H */
