#
# Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#
CUDA_HOME ?= /usr/local/cuda
PREFIX ?= /usr/local
VERBOSE ?= 0
DEBUG ?= 0

CUDA_LIB ?= $(CUDA_HOME)/lib64
CUDA_INC ?= $(CUDA_HOME)/include
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDARTLIB ?= cudart

CUDA_VERSION = $(strip $(shell which $(NVCC) >/dev/null && $(NVCC) --version | grep release | sed 's/.*release //' | sed 's/\,.*//'))
CUDA_MAJOR = $(shell echo $(CUDA_VERSION) | cut -d "." -f 1)
CUDA_MINOR = $(shell echo $(CUDA_VERSION) | cut -d "." -f 2)

# Better define NVCC_GENCODE in your environment to the minimal set
# of archs to reduce compile time.
ifeq ($(shell test "0$(CUDA_MAJOR)" -ge 13; echo $$?),0)
# Add Blackwell but drop Pascal & Volta support if we're using CUDA13.0 or above
NVCC_GENCODE ?= -gencode=arch=compute_75,code=sm_75 \
		-gencode=arch=compute_80,code=sm_80 \
		-gencode=arch=compute_90,code=sm_90 \
		-gencode=arch=compute_100,code=sm_100 \
		-gencode=arch=compute_120,code=sm_120 \
		-gencode=arch=compute_120,code=compute_120
else ifeq ($(shell test "0$(CUDA_MAJOR)" -eq 12 -a "0$(CUDA_MINOR)" -ge 8; echo $$?),0)
# Include Blackwell support if we're using CUDA12.8 or above
NVCC_GENCODE ?= -gencode=arch=compute_60,code=sm_60 \
		-gencode=arch=compute_61,code=sm_61 \
		-gencode=arch=compute_70,code=sm_70 \
		-gencode=arch=compute_80,code=sm_80 \
		-gencode=arch=compute_90,code=sm_90 \
		-gencode=arch=compute_100,code=sm_100 \
		-gencode=arch=compute_120,code=sm_120 \
		-gencode=arch=compute_120,code=compute_120
else ifeq ($(shell test "0$(CUDA_MAJOR)" -ge 12; echo $$?),0)
NVCC_GENCODE ?= -gencode=arch=compute_60,code=sm_60 \
                -gencode=arch=compute_61,code=sm_61 \
                -gencode=arch=compute_70,code=sm_70 \
		-gencode=arch=compute_80,code=sm_80 \
		-gencode=arch=compute_90,code=sm_90 \
		-gencode=arch=compute_90,code=compute_90
else ifeq ($(shell test "0$(CUDA_MAJOR)" -ge 11; echo $$?),0)
NVCC_GENCODE ?= -gencode=arch=compute_60,code=sm_60 \
                -gencode=arch=compute_61,code=sm_61 \
                -gencode=arch=compute_70,code=sm_70 \
		-gencode=arch=compute_80,code=sm_80 \
		-gencode=arch=compute_80,code=compute_80
else
NVCC_GENCODE ?= -gencode=arch=compute_35,code=sm_35 \
                -gencode=arch=compute_50,code=sm_50 \
                -gencode=arch=compute_60,code=sm_60 \
                -gencode=arch=compute_61,code=sm_61 \
                -gencode=arch=compute_70,code=sm_70 \
                -gencode=arch=compute_70,code=compute_70
endif

NVCUFLAGS  := -ccbin $(CXX) $(NVCC_GENCODE) -std=c++11
CXXFLAGS   := -std=c++11

LDFLAGS    := -L${CUDA_LIB} -lcudart -lrt
NVLDFLAGS  := -L${CUDA_LIB} -l${CUDARTLIB} -lrt

ifeq ($(DEBUG), 0)
NVCUFLAGS += -O3 -g
CXXFLAGS  += -O3 -g
else
NVCUFLAGS += -O0 -G -g
CXXFLAGS  += -O0 -g -ggdb3
endif

ifneq ($(VERBOSE), 0)
NVCUFLAGS += -Xcompiler -Wall,-Wextra,-Wno-unused-parameter
else
.SILENT:
endif
