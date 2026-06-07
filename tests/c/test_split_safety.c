/*************************************************************************
 * Unit tests for NCCL_TESTS_SPLIT* environment variable safety
 *
 * Tests verify that:
 *   1. strtoul(s, NULL, 16) silently fails on garbage input (adversarial)
 *   2. strtoul overflow is undetectable without errno (adversarial)
 *   3. Division by zero delivers SIGFPE (adversarial, fork-based)
 *   4. parseInt "0b" prefix has endptr comparison bug (adversarial)
 *   5. Safe hex parsing with endptr detects all error cases
 *   6. Division guard prevents crash on color==0
 *   7. Source files have been patched (source verification)
 *
 * The source verification test (test_source_verified) intentionally FAILS
 * before the fix is applied, demonstrating the bugs exist in unfixed code.
 *
 * Compile: gcc -Wall -Wextra -Wno-format-truncation -g -std=c99 -fPIC \
 *          -D_GNU_SOURCE -o test_split_safety test_split_safety.c -lpthread
 * Run:     ./test_split_safety
 *************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>
#include <errno.h>
#include <ctype.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>

/* =========================================================================
 * Test framework (matches NCCL tests/c/ pattern)
 * ========================================================================= */

#define TEST_ASSERT(condition, message)                                        \
  do {                                                                         \
    if (!(condition)) {                                                        \
      printf("  FAIL: %s - %s\n", __func__, message);                         \
      return 0;                                                                \
    }                                                                          \
  } while (0)

#define TEST_PASS()                                                            \
  do {                                                                         \
    printf("  PASS: %s\n", __func__);                                          \
    return 1;                                                                  \
  } while (0)

/* =========================================================================
 * Helpers
 * ========================================================================= */

static char *read_file(const char *path) {
  FILE *f = fopen(path, "r");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  long len = ftell(f);
  if (len <= 0) { fclose(f); return NULL; }
  fseek(f, 0, SEEK_SET);
  char *buf = (char *)malloc(len + 1);
  if (!buf) { fclose(f); return NULL; }
  size_t n = fread(buf, 1, len, f);
  buf[n] = '\0';
  fclose(f);
  return buf;
}

/*
 * Copy of the current (buggy) parseInt from common.cu.
 * It's static and behind #ifdef MPI_SUPPORT, so we reproduce it here
 * to test its behavior directly.
 */
static int parseInt_buggy(char *s, int *num) {
  char *p = NULL;
  if (!s || !num)
    return 0;
  while (*s && isspace(*s)) ++s;
  if (!*s) return 0;

  if (strncasecmp(s, "0b", 2) == 0)
    *num = (int)strtoul(s + 2, &p, 2);
  else
    *num = (int)strtoul(s, &p, 0);

  if (p == s)
    return 0;
  return 1;
}

/* =========================================================================
 * Test 1: Source verification
 * INTENTIONALLY FAILS before the fix is applied.
 * ========================================================================= */

int test_source_verified(void) {
  int all_ok = 1;
  char msg[512];

  const char *common_path = "../../src/common.cu";
  char *common_src = read_file(common_path);
  if (!common_src) {
    snprintf(msg, sizeof(msg), "Cannot read %s (run from tests/c/)", common_path);
    printf("  SKIP: %s - %s\n", __func__, msg);
    return 0;
  }

  /* Check 1: no raw strtoul with NULL endptr in SPLIT_MASK handling */
  if (strstr(common_src, "strtoul(splitMaskEnv, NULL,") != NULL) {
    snprintf(msg, sizeof(msg),
             "%s: strtoul(splitMaskEnv, NULL, 16) — no endptr validation (CWE-807)",
             common_path);
    printf("  FAIL: %s - %s\n", __func__, msg);
    all_ok = 0;
  }

  /* Check 2: division-by-zero guard exists for MOD/DIV operations */
  if (strstr(common_src, "color == 0") == NULL) {
    snprintf(msg, sizeof(msg),
             "%s: no 'color == 0' guard before proc %% color / proc / color (CWE-369)",
             common_path);
    printf("  FAIL: %s - %s\n", __func__, msg);
    all_ok = 0;
  }

  /* Check 3: parseInt uses errno check for overflow detection */
  if (strstr(common_src, "errno = 0") == NULL ||
      strstr(common_src, "errno == ERANGE") == NULL) {
    snprintf(msg, sizeof(msg),
             "%s: parseInt missing errno check for strtoul overflow",
             common_path);
    printf("  FAIL: %s - %s\n", __func__, msg);
    all_ok = 0;
  }

  free(common_src);

  if (!all_ok) return 0;
  TEST_PASS();
}

/* =========================================================================
 * Test 2: ADVERSARIAL — strtoul without endptr silently returns 0
 * Proves: NCCL_TESTS_SPLIT_MASK=xyz silently masks to 0
 * ========================================================================= */

int test_strtoul_no_endptr_exploit(void) {
  /*
   * The current code does:
   *   color = proc & strtoul(splitMaskEnv, NULL, 16);
   *
   * With garbage input "xyz", strtoul returns 0 — indistinguishable
   * from the valid input "0".  The caller cannot detect the error.
   */
  unsigned long garbage_result = strtoul("xyz", NULL, 16);
  unsigned long zero_result = strtoul("0", NULL, 16);

  TEST_ASSERT(garbage_result == 0,
    "strtoul(\"xyz\", NULL, 16) silently returns 0");
  TEST_ASSERT(zero_result == 0,
    "strtoul(\"0\", NULL, 16) also returns 0");
  TEST_ASSERT(garbage_result == zero_result,
    "EXPLOIT: garbage and valid '0' are indistinguishable without endptr");

  /* The safe alternative: use endptr to detect failure */
  char *end;
  strtoul("xyz", &end, 16);
  TEST_ASSERT(*end != '\0' || end == (char*)"xyz",
    "strtoul with endptr detects garbage: end points to unconsumed input");

  strtoul("ff", &end, 16);
  TEST_ASSERT(*end == '\0',
    "strtoul with endptr: valid hex fully consumed");

  TEST_PASS();
}

/* =========================================================================
 * Test 3: ADVERSARIAL — strtoul overflow without errno check
 * ========================================================================= */

int test_strtoul_overflow_exploit(void) {
  /*
   * strtoul on a value larger than ULONG_MAX returns ULONG_MAX
   * but without checking errno, the caller can't distinguish
   * overflow from a legitimately large value.
   */
  const char *huge = "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF"; /* way beyond ULONG_MAX */
  errno = 0;
  unsigned long result = strtoul(huge, NULL, 16);

  TEST_ASSERT(result == ULONG_MAX,
    "strtoul of overflow value returns ULONG_MAX");

  /* Without errno check, caller doesn't know this is overflow */
  int overflow_detected = (errno == ERANGE);
  TEST_ASSERT(overflow_detected,
    "only errno == ERANGE reveals the overflow");

  /* Reset and show a valid large value doesn't set ERANGE */
  errno = 0;
  strtoul("FF", NULL, 16);
  TEST_ASSERT(errno != ERANGE,
    "valid hex 'FF' does not set ERANGE");

  TEST_PASS();
}

/* =========================================================================
 * Test 4: ADVERSARIAL — division by zero delivers SIGFPE
 * Uses fork() to prove the crash without killing the test runner.
 * ========================================================================= */

int test_divzero_sigfpe_exploit(void) {
  int status;
  pid_t pid;

  /*
   * Prove: proc % 0 delivers SIGFPE.
   * This is exactly what happens with NCCL_TESTS_SPLIT=MOD0:
   *   parseInt("0", &color) succeeds with color=0
   *   color = proc % color → SIGFPE
   */
  pid = fork();
  if (pid == 0) {
    /* Child: perform the modulo that crashes */
    volatile int proc = 7;
    volatile int color = 0;
    volatile int result = proc % color;
    (void)result;
    _exit(0); /* should never reach here */
  }
  waitpid(pid, &status, 0);
  TEST_ASSERT(WIFSIGNALED(status),
    "EXPLOIT PROOF (MOD): child was killed by a signal");
  TEST_ASSERT(WTERMSIG(status) == SIGFPE,
    "EXPLOIT PROOF (MOD): proc %% 0 delivers SIGFPE — "
    "NCCL_TESTS_SPLIT=MOD0 crashes the MPI process");

  /*
   * Prove: proc / 0 also delivers SIGFPE.
   * This is NCCL_TESTS_SPLIT=DIV0.
   */
  pid = fork();
  if (pid == 0) {
    volatile int proc = 7;
    volatile int color = 0;
    volatile int result = proc / color;
    (void)result;
    _exit(0);
  }
  waitpid(pid, &status, 0);
  TEST_ASSERT(WIFSIGNALED(status),
    "EXPLOIT PROOF (DIV): child was killed by a signal");
  TEST_ASSERT(WTERMSIG(status) == SIGFPE,
    "EXPLOIT PROOF (DIV): proc / 0 delivers SIGFPE — "
    "NCCL_TESTS_SPLIT=DIV0 crashes the MPI process");

  TEST_PASS();
}

/* =========================================================================
 * Test 5: ADVERSARIAL — parseInt "0b" prefix endptr comparison bug
 * Proves: parseInt("0bxyz", &num) falsely succeeds with num=0
 * ========================================================================= */

int test_parseInt_0b_endptr_exploit(void) {
  /*
   * The current parseInt does:
   *   if (strncasecmp(s, "0b", 2) == 0)
   *     *num = (int)strtoul(s + 2, &p, 2);  // p points into s+2 area
   *   ...
   *   if (p == s)  // BUG: compares against s, not s+2
   *     return false;
   *
   * For input "0bxyz":
   *   s = "0bxyz", s+2 = "xyz"
   *   strtoul("xyz", &p, 2) → p = "xyz" (no valid binary digits)
   *   *num = 0
   *   p ("xyz") != s ("0bxyz") → return true!  BUG!
   *
   * This means NCCL_TESTS_SPLIT=MOD0bxyz silently gets color=0
   * and then proc % 0 = SIGFPE.
   */
  int num = -1;
  int result = parseInt_buggy("0bxyz", &num);

  printf("    parseInt_buggy(\"0bxyz\") = %s, num = %d\n",
         result ? "true" : "false", num);

  TEST_ASSERT(result == 1,
    "EXPLOIT: parseInt_buggy(\"0bxyz\") falsely returns true (endptr bug)");
  TEST_ASSERT(num == 0,
    "EXPLOIT: parseInt_buggy sets num=0 on garbage — feeds into MOD0 crash");

  /* For comparison, valid binary input works correctly */
  num = -1;
  result = parseInt_buggy("0b1010", &num);
  TEST_ASSERT(result == 1 && num == 10,
    "parseInt_buggy(\"0b1010\") correctly parses as 10");

  TEST_PASS();
}

/* =========================================================================
 * Test 6: Table-driven — safe hex parsing with endptr + errno
 * ========================================================================= */

int test_strtoul_hex_validation(void) {
  struct {
    const char *input;
    int expect_valid;
    unsigned long expect_value; /* only checked if valid */
    const char *label;
  } cases[] = {
    { "ff",         1, 0xff,       "lowercase hex" },
    { "FF",         1, 0xff,       "uppercase hex" },
    { "0",          1, 0,          "zero" },
    { "deadbeef",   1, 0xdeadbeef, "deadbeef" },
    { "1",          1, 1,          "one" },
    { "xyz",        0, 0,          "garbage" },
    { "",           0, 0,          "empty string" },
    { "ff garbage", 0, 0,          "partial parse (trailing garbage)" },
    { "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", 0, 0, "overflow" },
    { NULL, 0, 0, NULL }
  };

  for (int i = 0; cases[i].input != NULL; i++) {
    const char *input = cases[i].input;
    char *end;
    errno = 0;
    unsigned long val = strtoul(input, &end, 16);
    int valid = (end != input && *end == '\0' && errno != ERANGE);
    char msg[256];

    if (valid != cases[i].expect_valid) {
      snprintf(msg, sizeof(msg),
               "hex case '%s' (%s): expected %s, got %s",
               input, cases[i].label,
               cases[i].expect_valid ? "valid" : "invalid",
               valid ? "valid" : "invalid");
      TEST_ASSERT(0, msg);
    }
    if (valid && val != cases[i].expect_value) {
      snprintf(msg, sizeof(msg),
               "hex case '%s' (%s): expected 0x%lx, got 0x%lx",
               input, cases[i].label, cases[i].expect_value, val);
      TEST_ASSERT(0, msg);
    }
  }

  TEST_PASS();
}

/* =========================================================================
 * Test 7: Table-driven — division guard pattern
 * ========================================================================= */

int test_division_guard(void) {
  struct {
    int proc;
    int color;
    int expect_mod;    /* -1 means "should be guarded (skip)" */
    int expect_div;    /* -1 means "should be guarded (skip)" */
    const char *label;
  } cases[] = {
    { 7,  3,  1,  2,  "normal: 7 mod 3 = 1, 7 div 3 = 2" },
    { 8,  4,  0,  2,  "even split: 8 mod 4 = 0, 8 div 4 = 2" },
    { 0,  5,  0,  0,  "proc=0: 0 mod 5 = 0, 0 div 5 = 0" },
    { 7,  1,  0,  7,  "color=1: 7 mod 1 = 0, 7 div 1 = 7" },
    { 7,  0, -1, -1,  "DANGER: color=0 must be guarded" },
    { 0,  0, -1, -1,  "DANGER: both zero must be guarded" },
    { -1, 0, 0, 0, NULL }
  };

  for (int i = 0; cases[i].label != NULL; i++) {
    int proc = cases[i].proc;
    int color = cases[i].color;
    char msg[256];

    /* Test the guard pattern for MOD */
    if (color == 0) {
      /* Guard should prevent the operation */
      TEST_ASSERT(cases[i].expect_mod == -1,
        "color==0 cases must expect guard (-1)");
    } else {
      int mod_result = proc % color;
      if (mod_result != cases[i].expect_mod) {
        snprintf(msg, sizeof(msg),
                 "case '%s': %d %% %d expected %d, got %d",
                 cases[i].label, proc, color, cases[i].expect_mod, mod_result);
        TEST_ASSERT(0, msg);
      }
    }

    /* Test the guard pattern for DIV */
    if (color == 0) {
      TEST_ASSERT(cases[i].expect_div == -1,
        "color==0 cases must expect guard (-1)");
    } else {
      int div_result = proc / color;
      if (div_result != cases[i].expect_div) {
        snprintf(msg, sizeof(msg),
                 "case '%s': %d / %d expected %d, got %d",
                 cases[i].label, proc, color, cases[i].expect_div, div_result);
        TEST_ASSERT(0, msg);
      }
    }
  }

  TEST_PASS();
}

/* =========================================================================
 * Test runner
 * ========================================================================= */

typedef struct {
  const char *name;
  int (*func)(void);
  const char *description;
} TestCase;

static TestCase test_cases[] = {
    {"source-verified", test_source_verified,
     "Verify source uses validated strtoul and div-by-zero guards (FAILS before fix)"},
    {"strtoul-no-endptr-exploit", test_strtoul_no_endptr_exploit,
     "ADVERSARIAL: strtoul('xyz', NULL, 16) indistinguishable from '0'"},
    {"strtoul-overflow-exploit", test_strtoul_overflow_exploit,
     "ADVERSARIAL: strtoul overflow undetectable without errno"},
    {"divzero-sigfpe-exploit", test_divzero_sigfpe_exploit,
     "ADVERSARIAL: fork-proves proc%%0 and proc/0 deliver SIGFPE"},
    {"parseInt-0b-endptr-exploit", test_parseInt_0b_endptr_exploit,
     "ADVERSARIAL: parseInt('0bxyz') falsely succeeds with num=0"},
    {"strtoul-hex-validation", test_strtoul_hex_validation,
     "Table-driven safe hex parsing with endptr+errno"},
    {"division-guard", test_division_guard,
     "Table-driven division guard: color==0 skips, color>0 computes"},
    {NULL, NULL, NULL}
};

int main(int argc, char **argv) {
  const char *filter = NULL;
  int show_help = 0;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      show_help = 1;
    } else {
      filter = argv[i];
    }
  }

  if (show_help) {
    printf("Usage: %s [test-name]\n\n", argv[0]);
    printf("Available tests:\n");
    for (int i = 0; test_cases[i].name != NULL; i++) {
      printf("  %-35s %s\n", test_cases[i].name, test_cases[i].description);
    }
    printf("\nRun with no arguments to execute all tests.\n");
    return 0;
  }

  if (!filter) {
    filter = getenv("TEST_CASE");
  }

  printf("=== split env var safety tests ===\n\n");

  int passed = 0, total = 0;
  for (int i = 0; test_cases[i].name != NULL; i++) {
    if (filter && strcmp(filter, test_cases[i].name) != 0) {
      continue;
    }
    total++;
    passed += test_cases[i].func();
  }

  printf("\n%d/%d tests passed\n", passed, total);
  return (passed == total) ? 0 : 1;
}
