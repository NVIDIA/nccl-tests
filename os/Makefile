#
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#

include ../src/common.mk

.PHONY: build clean

BUILDDIR ?= ../build
DST_DIR := $(BUILDDIR)/os

ifneq ($(OS),Windows_NT)
  OS_SRC := linux.cc
else
  OS_SRC := windows.cc
endif

OBJ_FILES := $(OS_SRC:%.cc=${DST_DIR}/%.o)

build: ${DST_DIR}/libnccl_test_os.a

clean:
	rm -rf ${DST_DIR}

${DST_DIR}/%.o: %.cc os.h
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(CXX) $(CXXFLAGS) -I../src -o $@ -c $<

${DST_DIR}/libnccl_test_os.a: $(OBJ_FILES)
	@printf "Creating archive %-35s > %s\n" $^ $@
	@mkdir -p ${DST_DIR}
	ar rcs $@ $^
