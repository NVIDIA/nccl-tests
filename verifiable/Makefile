#
# Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#
include ../src/common.mk

.PHONY: all clean

BUILDDIR := $(abspath ../build)
NCCLDIR := $(BUILDDIR)
NVCUFLAGS += -I$(NCCLDIR)/include/ -I../include
DST_DIR := $(BUILDDIR)/verifiable

all: $(DST_DIR)/self_test

clean:
	rm -rf $(DST_DIR)

TEST_VERIFIABLE_SRCDIR := .
TEST_VERIFIABLE_BUILDDIR := $(DST_DIR)
include verifiable.mk

self_test: $(DST_DIR)/self_test

$(DST_DIR)/self_test: main.cu $(TEST_VERIFIABLE_LIBS)
	@printf "Linking  %s\n" $@
	@mkdir -p $(DST_DIR)
	$(NVCC) -o $@ $(NVCUFLAGS) $< -L$(TEST_VERIFIABLE_BUILDDIR) -lverifiable $(NVLDFLAGS) -Xlinker "-rpath=\$$ORIGIN"
