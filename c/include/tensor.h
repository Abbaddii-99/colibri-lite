#ifndef TENSOR_H
#define TENSOR_H

#include "model.h"

void rmsnorm(float *out, const float *x, const float *w, int D, float eps);
void layernorm(float *v, const float *w, const float *b, int n, float eps);
void softmax(float *x, int n);
void rope_interleave(float *v, int pos, const Cfg *c);
void qt_addrow(const QT *t, int row, float coef, float *acc);
void qt_matvec_rows(const QT *t, int r0, int n, const float *x, float *y);

#endif /* TENSOR_H */
