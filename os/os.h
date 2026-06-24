/*************************************************************************
 * Copyright (c) 2016-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef __NCCL_TEST_OS_H__
#define __NCCL_TEST_OS_H__

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

uint64_t ncclTestGetHostHash(const char* hostname);
int ncclTestGetHostname(char* name, size_t len);
int ncclTestGetPid();
int ncclTestStrcasecmp(const char* s1, const char* s2);
int ncclTestStrncasecmp(const char* s1, const char* s2, size_t n);
int ncclTestAsprintf(char** strp, const char* fmt, ...);
void ncclTestSetlinebuf(FILE* stream);

#ifdef __cplusplus
}
#endif

#if defined(NCCL_OS_LINUX)

#include <getopt.h>
#include <unistd.h>

#define NCCL_WEAK __attribute__((weak))

#elif defined(NCCL_OS_WINDOWS)

#include <windows.h>

#ifdef __cplusplus
extern "C" {
#endif

struct option {
  const char *name;
  int has_arg;
  int *flag;
  int val;
};

#define no_argument         0
#define required_argument   1
#define optional_argument   2

extern char *optarg;
extern int optind, opterr, optopt;

int getopt_long(int argc, char * const argv[], const char *optstring,
                const struct option *longopts, int *longindex);

#ifdef __cplusplus
}
#endif

#define NCCL_WEAK

#endif

#endif
