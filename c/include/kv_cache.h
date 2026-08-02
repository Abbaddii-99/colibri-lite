#ifndef KV_CACHE_H
#define KV_CACHE_H

#include "model.h"

void kv_alloc(Model *m, int max_t);
void kv_bind(Model *m, KVState *k);
float *step(Model *m, const int *ids, int S, int pos_base);
float *step_all(Model *m, const int *ids, int S, int pos_base);
void kv_disk_truncate(Model *m, int nrec);
void kv_disk_reset(Model *m);
void kv_disk_append(Model *m, const int *hist, int len);
int kv_disk_load(Model *m, int *hist, int maxctx);
double kv_pool_bytes(Model *m, int max_ctx);

#endif /* KV_CACHE_H */
