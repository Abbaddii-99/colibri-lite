#ifndef QUANT_H
#define QUANT_H

#include "model.h"

void matmul(float *y, const float *x, const float *W, int S, int I, int O);
void matmul_q(float *y, const float *x, const int8_t *q, const float *scale, int S, int I, int O);
void matmul_i4(float *y, const float *x, const uint8_t *q4, const float *scale, int S, int I, int O);
void matmul_i2(float *y, const float *x, const uint8_t *q2, const float *scale, int S, int I, int O);
void matmul_q_idot(float *y, const int8_t *xq, const float *sx, const int8_t *q, const float *scale, int S, int I, int O);
void matmul_i4_idot(float *y, const int8_t *xq, const float *sx, const uint8_t *q4, const float *scale, int S, int I, int O);
void quant_scratch(size_t xn, size_t sn, int8_t **xq, float **sx);
void matmul_qt(float *y, const float *x, QT *w, int S);
void quantize_rows(const float *w, int8_t *q, float *scale, int O, int I, int bits);
void pack_int4(const float *w, uint8_t *q4, float *scale, int O, int I, int bits);
void pack_int2(const float *w, uint8_t *q2, float *scale, int O, int I, int bits);
void qt_alloc(QT *t, int O, int I, int bits);
void qt_fill(QT *t, const float *w, int bits);

#endif /* QUANT_H */
