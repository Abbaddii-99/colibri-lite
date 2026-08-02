#ifndef CACHE_H
#define CACHE_H

#include "model.h"

void expert_load(Model *m, int layer, int eid, ESlot *s);
void expert_prefetch(Model *m, int layer, int eid);
void embed_row(Model *m, int tok, float *x);

/* PIPE: asynchronous expert loading */
void pipe_init(Model *m);
void pipe_dispatch(Model *m, int layer, const int *eids, int njobs);
void *pipe_worker(void *arg);
void pipe_wait(int q);

#endif /* CACHE_H */
