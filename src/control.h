/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#ifndef NCCL_TESTS_CONTROL_H_
#define NCCL_TESTS_CONTROL_H_

#include "common.h"
#include "util.h"
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <type_traits>
#include <vector>

// parse int for base 2/10/16, will ignore first whitespaces
static bool parseInt(const char *s, int *num) {
  char *p = NULL;
  if (!s || !num)
    return false;
  while (*s && isspace(*s)) ++s;
  if (!*s) return false;

  if (ncclTestStrncasecmp(s, "0b", 2) == 0)
    *num = (int)strtoul(s + 2, &p, 2);
  else
    *num = (int)strtoul(s, &p, 0);

  if (p == s)
    return false;
  return true;
}

static int getSplitColor(int proc) {
  int color = 0;
  const char *splitMaskEnv = NULL;
  if ((splitMaskEnv = getenv("NCCL_TESTS_SPLIT_MASK"))) {
    color = proc & strtoul(splitMaskEnv, NULL, 16);
  } else if ((splitMaskEnv = getenv("NCCL_TESTS_SPLIT"))) {
    if (
      (ncclTestStrncasecmp(splitMaskEnv, "AND", strlen("AND")) == 0 && parseInt(splitMaskEnv + strlen("AND"), &color)) ||
      (ncclTestStrncasecmp(splitMaskEnv, "&", strlen("&")) == 0 && parseInt(splitMaskEnv + strlen("&"), &color))
    )
        color = proc & color;
    if (
      (ncclTestStrncasecmp(splitMaskEnv, "OR", strlen("OR")) == 0 && parseInt(splitMaskEnv + strlen("OR"), &color)) ||
      (ncclTestStrncasecmp(splitMaskEnv, "|", strlen("|")) == 0 && parseInt(splitMaskEnv + strlen("|"), &color))
    )
        color = proc | color;
    if (
      (ncclTestStrncasecmp(splitMaskEnv, "MOD", strlen("MOD")) == 0 && parseInt(splitMaskEnv + strlen("MOD"), &color)) ||
      (ncclTestStrncasecmp(splitMaskEnv, "%", strlen("%")) == 0 && parseInt(splitMaskEnv + strlen("%"), &color))
    )
        color = proc % color;
    if (
      (ncclTestStrncasecmp(splitMaskEnv, "DIV", strlen("DIV")) == 0 && parseInt(splitMaskEnv + strlen("DIV"), &color)) ||
      (ncclTestStrncasecmp(splitMaskEnv, "/", strlen("/")) == 0 && parseInt(splitMaskEnv + strlen("/"), &color))
    )
        color = proc / color;
  }
  return color;
}

// MPI control starts before NCCL; native control binds to the benchmark communicators later.
#ifdef MPI_SUPPORT
#include "control_mpi.h"
#else
#include "control_env.h"
#endif

#endif
