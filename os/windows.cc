/*************************************************************************
 * Copyright (c) 2016-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "os.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

static uint64_t getHash(const char* string, size_t n) {
  uint64_t result = 5381;
  for (size_t c = 0; c < n; c++) {
    result = ((result << 5) + result) ^ string[c];
  }
  return result;
}

static bool getWindowsMachineGuid(char* guid, size_t len) {
  HKEY hKey;
  LONG result = RegOpenKeyExA(HKEY_LOCAL_MACHINE,
                              "SOFTWARE\\Microsoft\\Cryptography",
                              0,
                              KEY_READ,
                              &hKey);
  if (result != ERROR_SUCCESS) {
    return false;
  }

  DWORD dataSize = (DWORD)len;
  DWORD dataType;
  result = RegQueryValueExA(hKey,
                            "MachineGuid",
                            NULL,
                            &dataType,
                            (LPBYTE)guid,
                            &dataSize);

  RegCloseKey(hKey);

  return result == ERROR_SUCCESS && dataType == REG_SZ;
}

uint64_t ncclTestGetHostHash(const char* hostname) {
  char hostHash[1024];

  strncpy(hostHash, hostname, sizeof(hostHash));
  int offset = strlen(hostHash);

  char machineGuid[256];
  if (getWindowsMachineGuid(machineGuid, sizeof(machineGuid))) {
    strncpy(hostHash+offset, machineGuid, sizeof(hostHash)-offset-1);
  }

  hostHash[sizeof(hostHash)-1] = '\0';

  return getHash(hostHash, strlen(hostHash));
}

int ncclTestGetHostname(char* name, size_t len) {
  DWORD size = (DWORD)len;
  if (!GetComputerNameA(name, &size)) {
    return -1;
  }
  return 0;
}

int ncclTestGetPid() {
  return (int)GetCurrentProcessId();
}

int ncclTestStrcasecmp(const char* s1, const char* s2) {
  return _stricmp(s1, s2);
}

int ncclTestStrncasecmp(const char* s1, const char* s2, size_t n) {
  return _strnicmp(s1, s2, n);
}

int ncclTestAsprintf(char** strp, const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  int size = _vscprintf(fmt, args);
  va_end(args);
  if (size < 0) {
    return -1;
  }

  *strp = (char*)malloc(size + 1);
  if (*strp == NULL) {
    return -1;
  }

  va_start(args, fmt);
  int result = vsprintf_s(*strp, size + 1, fmt, args);
  va_end(args);
  return result;
}

void ncclTestSetlinebuf(FILE* stream) {
  setvbuf(stream, NULL, _IONBF, 0);
}

extern "C" {

int optind = 1;
int opterr = 1;
int optopt = 0;
char* optarg = NULL;

static char* nextchar = NULL;

int getopt_long(int argc, char* const argv[], const char* optstring,
                const struct option* longopts, int* longindex) {
  if (optind == 0) {
    optind = 1;
  }

  optarg = NULL;

  if (optind >= argc || argv[optind] == NULL) {
    return -1;
  }

  const char* arg = argv[optind];

  if (arg[0] == '-' && arg[1] == '-') {
    const char* name = arg + 2;
    const char* equals = strchr(name, '=');
    size_t name_len = equals ? (equals - name) : strlen(name);

    for (int i = 0; longopts[i].name != NULL; i++) {
      if (strncmp(longopts[i].name, name, name_len) == 0 &&
          strlen(longopts[i].name) == name_len) {
        if (longindex) {
          *longindex = i;
        }

        optind++;

        if (longopts[i].has_arg == required_argument) {
          if (equals) {
            optarg = (char*)(equals + 1);
          } else if (optind < argc) {
            optarg = argv[optind++];
          } else {
            if (opterr) {
              fprintf(stderr, "Option --%s requires an argument\n", longopts[i].name);
            }
            return '?';
          }
        } else if (longopts[i].has_arg == optional_argument) {
          if (equals) {
            optarg = (char*)(equals + 1);
          }
        }

        if (longopts[i].flag) {
          *(longopts[i].flag) = longopts[i].val;
          return 0;
        }
        return longopts[i].val;
      }
    }

    if (opterr) {
      fprintf(stderr, "Unrecognized option: %s\n", arg);
    }
    optind++;
    return '?';
  }

  if (arg[0] == '-' && arg[1] != '\0') {
    if (nextchar == NULL || *nextchar == '\0') {
      nextchar = (char*)(arg + 1);
    }

    char c = *nextchar++;
    const char* opt = strchr(optstring, c);

    if (opt == NULL) {
      optopt = c;
      if (opterr) {
        fprintf(stderr, "Invalid option: -%c\n", c);
      }
      if (*nextchar == '\0') {
        optind++;
        nextchar = NULL;
      }
      return '?';
    }

    if (opt[1] == ':') {
      if (*nextchar != '\0') {
        optarg = nextchar;
        optind++;
        nextchar = NULL;
      } else if (optind + 1 < argc) {
        optarg = argv[++optind];
        optind++;
        nextchar = NULL;
      } else {
        optopt = c;
        if (opterr) {
          fprintf(stderr, "Option -%c requires an argument\n", c);
        }
        optind++;
        nextchar = NULL;
        return '?';
      }
    } else if (*nextchar == '\0') {
      optind++;
      nextchar = NULL;
    }

    return c;
  }

  return -1;
}

}
