#pragma once

#include "nccl.h"
#include "nccl_device.h"
#include "args.h"

typedef struct {
  ncclDevComm dcomm;
  ncclDevResourceHandle hBuf;
} ginContext_t;

size_t ncclTestGinMaxBufferBytes(void);

void ncclTestGinDevCommCreate(ncclComm_t comm, const ginArgs_t* args, ginContext_t* ctx);
void ncclTestGinDevCommDestroy(ncclComm_t comm, ginContext_t* ctx);
