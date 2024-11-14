/**
 * Copyright (c) 2024, Scitix Tech PTE. LTD. All rights reserved.
 */

#ifndef __UCOMMD_H__
#define __UCOMMD_H__

#include <vector>
#include <string>

class Ucommd {
 public:
  Ucommd();
  ~Ucommd();

 public:
  int getLocalSize() const;

  int getNGpusPerProc() const;
  int getTimeoutSec() const;
  size_t getBytes() const;

  int getBw(int ngpus = -1);

 private:
  int get_nvlink_bw();
  int get_ib_bw();

 private:
  void _check_multi_node_via_ompi();
  void _check_sys_nv_devices();
  void _check_sys_ib_devices();
  void _get_node_name();

 private:
  int world_size_ = -1;
  int local_size_ = -1;
  int nnodes_ = -1;
  bool is_multi_node_ = false;
  std::string node_name_;
  std::vector<std::string> nvdevs_;
  std::vector<std::string> ibdevs_;
};

#endif
