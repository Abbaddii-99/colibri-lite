#ifndef PIPELINE_H
#define PIPELINE_H

#include "model.h"

void la_predict(Model *m, int target, const float *h, int kind);
void *pilot_worker(void *arg);
void pilot_prefetch(Model *m, int lnext, const float *x, int S);
void layer_forward(Model *m, Layer *l, int li, float *x, int S, int pos_base, float *nrm, float *tmp);
void layers_forward(Model *m, float *x, int S, int pos_base);

#endif /* PIPELINE_H */
