#ifndef ENGINE_H
#define ENGINE_H

#include "model.h"

void forward_all(Model *m, const int *ids, int S, int *pred);
void generate(Model *m, const int *prompt, int np, int n_new, int *out);
void profile_print(Model *m, double elapsed);
void run_replay(Model *m, const int *full, int nfull, int np);
void run_text(Model *m, const char *snap, const char *prompt, int ngen);
void run_score(Model *m, const char *path);
void run_chat(Model *m, const char *snap);
void run_serve(Model *m, const char *snap);
void repin_pass(Model *m);
int repin_pick(Model *m, RepinCand *out, int maxc);
void stats_dump(Model *m, const char *path);
void usage_save(Model *m);

#endif /* ENGINE_H */
