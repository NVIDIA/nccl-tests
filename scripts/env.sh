#!/bin/bash
export SIHPC_HOME=/usr/local/sihpc
export PATH=$SIHPC_HOME/bin:$PATH
export LD_LIBRARY_PATH=$SIHPC_HOME/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export OMPI_MCA_opal_prefix=$SIHPC_HOME
export OPAL_PREFIX=$SIHPC_HOME