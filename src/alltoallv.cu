#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include "cuda_runtime.h"
#include "common.h"

int CHECK = 0;

/**
 * @brief Parses the parameter file and stores the matrix data into the imbalancingFactors reference passed in. 
 * @param nranks The number of ranks in the test
 * @param imbalancingFactors The reference to the vector that will store the parsed data
 * @param filename The name of the parameter file to parse
**/
testResult_t parseParamFile(int nranks, std::vector<std::vector<double>> &imbalancingFactors, const char filename[PATH_MAX]){
  std::vector<std::vector<double>> paramFile_data;
  std::ifstream paramFile(filename);

  if (!paramFile.is_open()) {
    PRINT("\nUNABLE TO OPEN PARAMS FILE AT: %s\n", filename);
    return testInternalError;
  }

  std::string row;
  int rowidx = 0;
  while(std::getline(paramFile,row)){ //iterate over every row
    std::vector<double> values; //values from this line
    std::stringstream rowstream(row);
    std::string value;
    while(std::getline(rowstream,value,',')){ //go over the row and get each value  
      double dval = std::stod(value);
      if(dval<0 || dval>1) {
        PRINT("\nINVALID PARAMS FILE, PARAMETER OUT OF 0:1 RANGE, ROW NUMBER: %i \n", rowidx);
        return testInternalError;
      } //ensure that the value is between 0 and 1 (necessary for probability distribution)
      values.push_back(dval);
    }
    if(values.size()!=nranks) {
      PRINT("\nINVALID PARAMS FILE, ROW %i DOES NOT HAVE CORRECT NUMBER OF VALUES, HAS %lu ENTRIES, NEEDS %i ENTRIES\n", rowidx, values.size(), nranks);
      return testInternalError;
    }//ensure that this row has the right amount of values
    paramFile_data.push_back(values);
    rowidx++;
  }

  if(paramFile_data.size()!=nranks) {
    PRINT("\nINVALID PARAMS FILE, DOES NOT HAVE CORRECT NUMBER OF ROWS, HAS %i ROWS, NEEDS %i ROWS\n", paramFile_data.size(), nranks);
    return testInternalError;
  } //ensure we have the right amount of rows
  
  imbalancingFactors = paramFile_data; //store the data in the return variable
  return testSuccess;
} 
void AlltoAllvGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, int nranks) {
  *sendcount = (count/nranks)*nranks; //Total send count rounded to a multiple of ranks 
  *recvcount = (count/nranks)*nranks; //Total recv count rounded to a multiple of ranks
  *sendInplaceOffset = 0;
  *recvInplaceOffset = 0;
  *paramcount = (count/nranks); //Each rank can send a maximum of count/nranks data to each other rank
}

testResult_t AlltoAllvInitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place) {
  size_t maxchunk = args->nbytes / wordSize(type);
  int nranks = args->nProcs*args->nThreads*args->nGpus;
  //parse the param file
  std::vector<std::vector<double>> imbalancingFactors;
  testResult_t parseSuccess = parseParamFile(nranks, imbalancingFactors, args->setup_file);
  if(parseSuccess != testSuccess) return parseSuccess;
  for (int i=0; i<args->nGpus; i++) {
    CUDACHECK(cudaSetDevice(args->gpus[i]));
    CUDACHECK(cudaMemset(args->recvbuffs[i], 0, args->expectedBytes)); //zeroes out the receive buffer of each GPU with total size (recvcount*wordSize(type))
    CUDACHECK(cudaMemcpy(args->expected[i], args->recvbuffs[i], args->expectedBytes, cudaMemcpyDefault)); //copies the zeroed out receive buffer to the expected buffer
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i); //current rank
    void* data = in_place ? args->recvbuffs[i] : args->sendbuffs[i];
    TESTCHECK(InitData(data, maxchunk*nranks, 0, type, ncclSum, 33*rep + rank, 1, 0)); //initializes the sendbuffer data for this rank. Should be chunk size * nranks
    for (int j=0; j<nranks; j++) { 
      size_t partcount_mod = maxchunk * imbalancingFactors[j][rank]; //imbalance the count of data to initialize same way we do in the test
      TESTCHECK(InitData((char*)args->expected[i] + j*maxchunk*wordSize(type), partcount_mod, rank*maxchunk, type, ncclSum, 33*rep + j, 1, 0));
    }
    CUDACHECK(cudaDeviceSynchronize());
  }
  // We don't support in-place alltoallv
  args->reportErrors = in_place ? 0 : 1;
  return testSuccess;
}

void AlltoAllvGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * nranks * typesize) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = ((double)(nranks-1))/((double)(nranks));
  *busBw = baseBw * factor;
}

testResult_t AlltoAllvRunColl(void* sendbuff, void* recvbuff, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, struct threadArgs* args) {
  int nRanks, myRank;
  NCCLCHECK(ncclCommCount(comm, &nRanks));
  NCCLCHECK(ncclCommUserRank(comm, &myRank));
  std::vector<std::vector<double>> imbalancingFactors; 
  testResult_t parseSuccess = parseParamFile(nRanks, imbalancingFactors, args->setup_file); //parse the param file
  if(parseSuccess != testSuccess) return parseSuccess;
  size_t rankOffset = count * wordSize(type);

#if NCCL_MAJOR < 2 || NCCL_MINOR < 7
  printf("NCCL 2.7 or later is needed for alltoallv. This test was compiled with %d.%d.\n", NCCL_MAJOR, NCCL_MINOR);
  return testNcclError;
#else
  NCCLCHECK(ncclGroupStart());
  for (int r=0; r<nRanks; r++) {
    // int count_mod = (count-myRank-r-1) % count; //modify the count variable to to be strictly less than count, but depend on both the peer rank and the sending rank
    if(myRank>imbalancingFactors.size()){
      PRINT("\nmyRank is greater than imbalancingFactors.size(), %i\n", myRank);
      return testInternalError;
    } else if (r > imbalancingFactors[myRank].size()) {
        PRINT("\nr is greater than imbalancingFactors[myRank].size(), %i\n", r);
        return testInternalError;
    }
    unsigned long send_count_mod = count * imbalancingFactors[myRank][r]; 
    unsigned long recv_count_mod = count * imbalancingFactors[r][myRank]; 
    NCCLCHECK(ncclSend(((char*)sendbuff)+r*rankOffset, send_count_mod, type, r, comm, stream));
    NCCLCHECK(ncclRecv(((char*)recvbuff)+r*rankOffset, recv_count_mod, type, r, comm, stream));
  }


  NCCLCHECK(ncclGroupEnd());
  return testSuccess;
#endif
}

struct testColl AlltoAllvTest = {
  "AlltoAllV",
  AlltoAllvGetCollByteCount,
  AlltoAllvInitData,
  AlltoAllvGetBw,
  AlltoAllvRunColl
};

void AlltoAllvGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  AlltoAllvGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, nranks);
}

testResult_t AlltoAllvRunTest(struct threadArgs* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  args->collTest = &AlltoAllvTest;
  ncclDataType_t *run_types;
  const char **run_typenames;
  int type_count;
  if ((int)type != -1) {
    type_count = 1;
    run_types = &type;
    run_typenames = &typeName;
  } else {
    type_count = test_typenum;
    run_types = test_types;
    run_typenames = test_typenames;
  }

  for (int i=0; i<type_count; i++) {
      TESTCHECK(TimeTest(args, run_types[i], run_typenames[i], (ncclRedOp_t)0, "none", -1));
  }
  return testSuccess;
}

struct testEngine AlltoAllvEngine = {
  AlltoAllvGetBuffSize,
  AlltoAllvRunTest
};

#pragma weak ncclTestEngine=AlltoAllvEngine
