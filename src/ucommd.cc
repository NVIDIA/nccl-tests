/**
 * Copyright (c) 2024, Scitix Tech PTE. LTD. All rights reserved.
 */

#include <unistd.h>
#include <dirent.h>
#include <string.h>

#include <algorithm>
#include <vector>
#include <string>
#include <fstream>

#include "ucommd.h"

Ucommd::Ucommd() {
  (void)_get_node_name();
  (void)_check_multi_node_via_ompi();
  (void)_check_sys_nv_devices();
  (void)_check_sys_ib_devices();
}

Ucommd::~Ucommd() {
  nvdevs_.clear();
  ibdevs_.clear();
}

void Ucommd::_check_multi_node_via_ompi() {
  const auto world_size_env = std::getenv("OMPI_COMM_WORLD_SIZE");
  if (world_size_env == nullptr) return;
  world_size_ = std::strtol(world_size_env, nullptr, 10);
  const auto local_size_env = std::getenv("OMPI_COMM_WORLD_LOCAL_SIZE");
  if (local_size_env == nullptr) return;
  local_size_ = std::strtol(local_size_env, nullptr, 10);

  // assume homogeneous mpirun
  nnodes_ = world_size_ / local_size_;
  is_multi_node_ = (nnodes_ > 1);
}

void Ucommd::_check_sys_nv_devices() {
  DIR* dir;
  dir = opendir("/sys/bus/pci/drivers/nvidia");
  if (dir) {
    struct dirent *entry;
    while ((entry = readdir(dir))) {
      if (entry->d_name[0] != '0') continue;
      const auto nvdev = std::string(entry->d_name);
      auto dev_class = std::ifstream(
          std::string("/sys/bus/pci/drivers/nvidia/") + nvdev + "/class");
      if (dev_class.is_open()) {
        char dclass[16] = {0};
        dev_class.getline(dclass, 16);
        if (dev_class.good() &&
           (std::string("0x030200").compare(dclass) == 0 ||
            std::string("0x030000").compare(dclass) == 0)) {
          nvdevs_.push_back(nvdev);
        }
        dev_class.close();
      }
    }
    closedir(dir);
    std::sort(nvdevs_.begin(), nvdevs_.end());
  }
}

void Ucommd::_check_sys_ib_devices() {
  DIR* dir;
  dir = opendir("/sys/class/infiniband");
  if (dir) {
    struct dirent *entry;
    while ((entry = readdir(dir))) {
      if ((strcmp(entry->d_name, ".") == 0) ||
          (strcmp(entry->d_name, "..") == 0)) {
        continue;
      }
      const auto ibdev = std::string(entry->d_name);
      if ([&ibdev] {
          bool is_ib = false;
          auto node_type = std::ifstream(
              std::string("/sys/class/infiniband/") + ibdev + "/node_type");
          if (node_type.is_open()) {
            char ntype = node_type.get();
            if (node_type.good()) is_ib = '1' <= ntype && ntype <= '3';
            node_type.close();
          }
          return is_ib;
        }() &&
        [&ibdev] {
          bool is_cx6 = false;
          auto hca_type = std::ifstream(
              std::string("/sys/class/infiniband/") + ibdev + "/hca_type");
          if (hca_type.is_open()) {
            char htype[8] = {0};
            hca_type.getline(htype, 8);
            if (hca_type.good()) {
              is_cx6 = std::string("MT4123").compare(htype) == 0 ||
                       std::string("MT4125").compare(htype) == 0 ||
                       std::string("MT4129").compare(htype) == 0 ||
                       std::string("MT4131").compare(htype) == 0 ||
                       std::string("MT4124").compare(htype) == 0;
            }
            hca_type.close();
          }
          return is_cx6;
        }() &&
        [&ibdev, this] {
          bool port_active = false;
          auto port_state = std::ifstream(
              std::string("/sys/class/infiniband/") + ibdev + "/ports/1/state");
          if (port_state.is_open()) {
            char state = port_state.get();
            if (port_state.good()) {
              port_active = state == '4';
            }
            port_state.close();
          }
          if (!port_active) {
            printf("[%s] %s: port not active or unable to get port state\n",
                node_name_.c_str(), ibdev.c_str());
          }
          return port_active;
        }() &&
        [&ibdev, this] {
          bool link_up = false;
          auto phys_state = std::ifstream(
              std::string("/sys/class/infiniband/") + ibdev + "/ports/1/phys_state");
          if (phys_state.is_open()) {
            char state = phys_state.get();
            if (phys_state.good()) {
              link_up = state == '5';
            }
            phys_state.close();
          }
          if (!link_up) {
            printf("[%s] %s: phys link not up or unable to get phys state\n",
                node_name_.c_str(), ibdev.c_str());
          }
          return link_up;
        }()) {
        ibdevs_.push_back(ibdev);
      }
    }
    closedir(dir);
    std::sort(ibdevs_.begin(), ibdevs_.end());
  }
}

void Ucommd::_get_node_name() {
  const auto node_name_env = std::getenv("NODE_NAME");
  if (node_name_env && node_name_env[0]) {
    node_name_.assign(node_name_env);
  } else {
    char hostname[128] = {0};
    if (!gethostname(hostname, 128)) {
      node_name_.assign(hostname);
    } else {
      node_name_.assign("unknown");
    }
  }
}

int Ucommd::getLocalSize() const {
  return local_size_;
}

int Ucommd::getNGpusPerProc() const {
  return (is_multi_node_ || local_size_ > 1) ? 1 : (int)nvdevs_.size();
}

size_t Ucommd::getBytes() const {
  return !is_multi_node_ ? 1UL << 32 :
      world_size_ > 1024 ? ((size_t)world_size_) << 24 :
      local_size_ > 4 ? ((size_t)world_size_) << 25 :
      local_size_ > 1 ? ((size_t)world_size_) << 26 :
      ((size_t)world_size_) << 27;
}

int Ucommd::getTimeoutSec() const {
  return 600;
}

int Ucommd::getBw(int ngpus) {
  return is_multi_node_ ? get_ib_bw() :
      (ngpus > 1 || local_size_ > 1) ? get_nvlink_bw() : -1;
}

int Ucommd::get_nvlink_bw() {
  int bw = -1;
  if (!nvdevs_.empty()) {
    auto dev_id = std::ifstream(
        std::string("/sys/bus/pci/drivers/nvidia/") + nvdevs_.at(0) + "/device");
    if (dev_id.is_open()) {
      char device[16] = {0};
      dev_id.getline(device, 16);
      if (dev_id.good()) {
        if (std::string("0x2330").compare(device) == 0) {
          bw = 450 * 3 / 4;
        } else
        if (std::string("0x20b0").compare(device) == 0 ||
            std::string("0x20b2").compare(device) == 0 ||
            std::string("0x20b3").compare(device) == 0) {
          bw = 300 * 2 / 3;
        } else
        if (std::string("0x20f3").compare(device) == 0 ||
            std::string("0x20bd").compare(device) == 0) {
          bw = 200 * 2 / 3;
        }
      }
    }
  }
  return bw;
}

int Ucommd::get_ib_bw() {
  int bw = -1;
  if (!ibdevs_.empty()) {
    int rate = 0;
    auto port_rate = std::ifstream(
        std::string("/sys/class/infiniband/") + ibdevs_.at(0) + "/ports/1/rate");
    if (port_rate.is_open()) {
      char c;
      while ((c = port_rate.get()) && ('0' <= c && c <= '9')) {
        rate = rate * 10 + c - '0';
      }
      port_rate.close();
    }
    bw = rate * 3 / 32;

    // for DP AllReduce only ...
    auto nnics = ibdevs_.size();
    if (local_size_ == 2) {
      bw = nnics > 1 ? bw * 2 : bw;
    } else
    if (local_size_ == 4) {
      bw = nnics > 3 ? bw * 4 : nnics > 1 ? bw * 2 : bw;
    } else
    if (local_size_ == 8) {
      bw *= nnics;
      const char* mask_env = getenv("NCCL_TESTS_SPLIT_MASK");
      if (mask_env) {
        auto mask = std::strtol(mask_env, nullptr, 10);
        if (mask == 7 || mask == 3 || mask == 1) {
          bw /= (mask+1);
        } // else ???
      }
    }
  }
  return bw;
}
