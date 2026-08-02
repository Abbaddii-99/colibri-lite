#ifndef LOAD_H
#define LOAD_H

#include "model.h"

void load_cfg(Cfg *c, const char *snap);
void qt_from_disk(Model *m, const char *name, int O, int I, int bits, int drop, QT *t);
QT qt_load(Model *m, const char *name, int O, int I, int bits);
float *ld(Model *m, const char *name);
void model_init(Model *m, const char *snap, int cap, int ebits, int dbits);
void model_free(Model *m);

#endif /* IO_H */
