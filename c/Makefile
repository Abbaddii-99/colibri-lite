CC      = gcc
CFLAGS  = -O3 -march=native -std=c11
CFLAGS  += -Wall -Wextra -Wno-unused-parameter -Wno-unused-function
CFLAGS  += -Wno-unknown-pragmas -Wno-misleading-indentation
CFLAGS  += -D_FILE_OFFSET_BITS=64
INCS    := -Iinclude
SRCDIR  := src
OBJDIR  := obj
TARGET  := glm

SRCS    := $(filter-out $(SRCDIR)/bench.c $(SRCDIR)/moe_debug.c, $(wildcard $(SRCDIR)/*.c))
OBJS    := $(patsubst $(SRCDIR)/%.c,$(OBJDIR)/%.o,$(SRCS))
DEPS    := $(OBJS:.o=.d)

# CUDA support (optional): make CUDA=1 or CUDA_DEVICES="0"
ifdef CUDA
  NVCC      ?= nvcc
  NVCCFLAGS  = -O3 -arch=native -DCOLI_CUDA $(INCS) --compiler-options=-fPIC
  CU_SRCS   := $(wildcard $(SRCDIR)/*.cu)
  CU_OBJS   := $(patsubst $(SRCDIR)/%.cu,$(OBJDIR)/%.o,$(CU_SRCS))
  CFLAGS    += -DCOLI_CUDA
  LDFLAGS   += -lcuda -lcudart
  OBJS      += $(CU_OBJS)
endif

ifeq ($(OS),Windows_NT)
  MKDIR  := if not exist $(OBJDIR) mkdir $(OBJDIR)
  RM     := del /F /Q
else
  MKDIR  := mkdir -p $(OBJDIR)
  RM     := rm -rf
endif

.PHONY: all clean test bench cuda

all: $(TARGET)

bench: bench.exe

bench.exe: $(SRCDIR)/bench.c
	$(CC) $(CFLAGS) -Wno-alloc-size-larger-than $(INCS) $< -o $@ -lm -lpthread -fopenmp

cuda: CUDA=1
cuda: all

$(OBJDIR):
	$(MKDIR)

$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(INCS) -MMD -MP -c $< -o $@

ifdef CUDA
$(OBJDIR)/%.o: $(SRCDIR)/%.cu | $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
endif

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(INCS) $^ -o $@ $(LDFLAGS) -lm -lpthread

test: $(TARGET)
	python tests/smoke_test.py
	python tests/regression_test.py
	python tests/edge_test.py

clean:
	$(RM) $(OBJDIR)
	$(RM) $(TARGET)

-include $(DEPS)
