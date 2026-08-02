#ifndef SPEC_H
#define SPEC_H

#include "model.h"

int ngram_draft(const int *ids, int len, int G, int *draft);
int mtp_argmax(const float *lo, int V);
int mtp_draft(Model *m, int next_tok, int kv, int G, int *draft);
void mtp_absorb(Model *m, const int *next_ids, const float *x, int S, int pos_base);
int spec_decode(Model *m, int *all, int kv, int n_new, int eos, float *logit, int *out, int *nout);

/* Grammar */
typedef struct { int *dst, n, cap; } EmitStore;
typedef struct { void *T; struct Model *m; double t0; int count; int quiet; } EmitStream;
void grammar_setup(void *T);
void grammar_reset(void);
void gr_feed(int t);
int grammar_draft(int *draft, int cap);
void emit_store(int t, void *ud);
void emit_stream(int t, void *ud);

#endif /* SPEC_H */
