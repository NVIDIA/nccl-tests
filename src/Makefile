#
# Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#
include common.mk

# Set to 1 to enable MPI support (multi-process/multi-node)
MPI ?= 0
# e.g. Set to _mpi when using MPI=1
NAME_SUFFIX ?=
# Set to 1 to create and use libverifiable.so to reduce binary size
DSO ?= 0

.PHONY: build clean

BUILDDIR ?= ../build
ifneq ($(NCCL_HOME), "")
NVCUFLAGS += -I$(NCCL_HOME)/include/
NVLDFLAGS += -L$(NCCL_HOME)/lib
endif

ifeq ($(MPI), 1)
NVCUFLAGS += -DMPI_SUPPORT -I$(MPI_HOME)/include
NVLDFLAGS += -L$(MPI_HOME)/lib -L$(MPI_HOME)/lib64 -lmpi
endif
ifeq ($(MPI_IBM),1)
NVCUFLAGS += -DMPI_SUPPORT
NVLDFLAGS += -lmpi_ibm
endif
LIBRARIES += nccl
NVLDFLAGS += $(LIBRARIES:%=-l%)

DST_DIR := $(BUILDDIR)
SRC_FILES := $(wildcard *.cu)
OBJ_FILES := $(SRC_FILES:%.cu=${DST_DIR}/%.o)
BIN_FILES_LIST := all_reduce all_gather broadcast reduce_scatter reduce alltoall scatter gather sendrecv hypercube
BIN_FILES := $(BIN_FILES_LIST:%=${DST_DIR}/%_perf${NAME_SUFFIX})

build: ${BIN_FILES}

clean:
	rm -rf ${DST_DIR}

TEST_VERIFIABLE_SRCDIR := ../verifiable
TEST_VERIFIABLE_BUILDDIR := $(BUILDDIR)/verifiable
include ../verifiable/verifiable.mk

.PRECIOUS: ${DST_DIR}/%.o

${DST_DIR}/%.o: %.cu common.h util.h $(TEST_VERIFIABLE_HDRS)
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) -c $<

${DST_DIR}/%$(NAME_SUFFIX).o: %.cu common.h util.h $(TEST_VERIFIABLE_HDRS)
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) -c $<

${DST_DIR}/timer.o: timer.cc timer.h
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(CXX) $(CXXFLAGS) -o $@ -c $<

ifeq ($(DSO), 1)
${DST_DIR}/%_perf$(NAME_SUFFIX): ${DST_DIR}/%.o ${DST_DIR}/common$(NAME_SUFFIX).o ${DST_DIR}/util$(NAME_SUFFIX).o ${DST_DIR}/timer.o $(TEST_VERIFIABLE_LIBS)
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $^ -L$(TEST_VERIFIABLE_BUILDDIR) -lverifiable ${NVLDFLAGS} -Xlinker "--enable-new-dtags" -Xlinker "-rpath,\$$ORIGIN:\$$ORIGIN/verifiable"
else
${DST_DIR}/%_perf$(NAME_SUFFIX):${DST_DIR}/%.o ${DST_DIR}/common$(NAME_SUFFIX).o ${DST_DIR}/util$(NAME_SUFFIX).o ${DST_DIR}/timer.o $(TEST_VERIFIABLE_OBJS)
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(NVCC) -o $@ $(NVCUFLAGS) $^ ${NVLDFLAGS}
endif

clean_intermediates:
	rm -f ${DST_DIR}/*.o $(TEST_VERIFIABLE_OBJS)

