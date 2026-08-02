#ifndef ATTENTION_H
#define ATTENTION_H

#include "model.h"

void attention(Model *m, Layer *l, int layer, float *x, int S, int pos_base, float *out);

#endif /* ATTENTION_H */
