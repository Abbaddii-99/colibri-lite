#ifndef SAMPLING_H
#define SAMPLING_H

#include "model.h"

void dist_build(const float *lo, int V);
int dist_sample(int V, int ban);
int pick_tok(const float *lo, int V, int ban);
void stops_arm(const Cfg *c, int tok_eos);

#endif /* SAMPLING_H */
