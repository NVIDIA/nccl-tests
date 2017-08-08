#
# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#
# See LICENCE.txt for license information
#

.PHONY : all clean

default : src.build

TARGETS=src

all:   ${TARGETS:%=%.build}
clean: ${TARGETS:%=%.clean}

%.build:
	${MAKE} -C $* build

%.clean:
	${MAKE} -C $* clean
