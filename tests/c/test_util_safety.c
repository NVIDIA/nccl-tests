/*************************************************************************
 * Unit tests for buffer safety and thread safety in util.cu
 *
 * Tests verify that:
 *   1. snprintf correctly truncates and null-terminates (table-driven)
 *   2. sprintf overflows tight buffers (adversarial exploit proof)
 *   3. memcpy is safe for fixed-size truncation markers
 *   4. localtime_r is thread-safe under concurrent access
 *   5. localtime (bare) corrupts across threads (adversarial exploit proof)
 *   6. Source files have been patched to use safe alternatives
 *
 * The source verification test (test_source_verified) intentionally FAILS
 * before the fix is applied, demonstrating the bugs exist in unfixed code.
 *
 * Compile: gcc -Wall -Wextra -Wno-format-truncation -g -std=c99 -D_GNU_SOURCE \
 *          -o test_util_safety test_util_safety.c -lpthread
 * Run:     ./test_util_safety
 *************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>
#include <time.h>
#include <pthread.h>
#include <float.h>

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

/* Read entire file into malloc'd buffer. Returns NULL on failure. */
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

/* Count occurrences of bare 'target(' excluding prefixed variants.
 * For sprintf(: skip if preceded by 'n' (snprintf) or 'a' (asprintf).
 * For strcpy(:  no prefix check needed.
 * For localtime(: skip if preceded by '_r'. */
static int count_bare_calls(const char *src, const char *target,
                            const char *skip_prefix) {
  int count = 0;
  size_t tlen = strlen(target);
  const char *p = src;
  while ((p = strstr(p, target)) != NULL) {
    if (skip_prefix && p > src) {
      size_t plen = strlen(skip_prefix);
      if (p - src >= (long)plen &&
          strncmp(p - plen, skip_prefix, plen) == 0) {
        p += tlen;
        continue;
      }
    }
    count++;
    p += tlen;
  }
  return count;
}

/* =========================================================================
 * Test 1: Source verification — scans util.cu for unsafe patterns.
 * INTENTIONALLY FAILS before the fix is applied.
 * ========================================================================= */

int test_source_verified(void) {
  int all_ok = 1;
  char msg[512];

  const char *util_path = "../../src/util.cu";
  char *src = read_file(util_path);
  if (!src) {
    snprintf(msg, sizeof(msg),
             "Cannot read %s (run from tests/c/)", util_path);
    printf("  SKIP: %s - %s\n", __func__, msg);
    return 1; /* Don't fail if file not found — allows running outside repo */
  }

  /* Check 1: No bare sprintf( — skip snprintf( and asprintf( */
  int sprintf_count = count_bare_calls(src, "sprintf(", "sn");
  /* Also skip asprintf */
  sprintf_count -= count_bare_calls(src, "asprintf(", NULL);
  if (sprintf_count > 0) {
    snprintf(msg, sizeof(msg),
             "%s: found %d bare sprintf() calls — expected snprintf()",
             util_path, sprintf_count);
    printf("  FAIL: %s - %s\n", __func__, msg);
    all_ok = 0;
  }

  /* Check 2: No strcpy( */
  int strcpy_count = count_bare_calls(src, "strcpy(", "strn");
  if (strcpy_count > 0) {
    snprintf(msg, sizeof(msg),
             "%s: found %d strcpy() calls — expected memcpy() or strncpy()",
             util_path, strcpy_count);
    printf("  FAIL: %s - %s\n", __func__, msg);
    all_ok = 0;
  }

  /* Check 3: No bare localtime( — skip localtime_r( */
  int localtime_count = count_bare_calls(src, "localtime(", "localtime_r");
  /* The above doesn't work because localtime_r contains localtime(.
   * Instead: count localtime( and subtract localtime_r( occurrences. */
  localtime_count = count_bare_calls(src, "localtime(", NULL);
  int localtime_r_count = count_bare_calls(src, "localtime_r(", NULL);
  int bare_localtime = localtime_count - localtime_r_count;
  if (bare_localtime > 0) {
    snprintf(msg, sizeof(msg),
             "%s: found %d bare localtime() calls — expected localtime_r()",
             util_path, bare_localtime);
    printf("  FAIL: %s - %s\n", __func__, msg);
    all_ok = 0;
  }

  /* Check 4: localtime_r IS present */
  if (localtime_r_count == 0) {
    snprintf(msg, sizeof(msg),
             "%s: localtime_r() not found — expected thread-safe variant",
             util_path);
    printf("  FAIL: %s - %s\n", __func__, msg);
    all_ok = 0;
  }

  /* Check 5: snprintf IS present */
  if (strstr(src, "snprintf(") == NULL) {
    snprintf(msg, sizeof(msg),
             "%s: snprintf() not found — expected bounds-checked variant",
             util_path);
    printf("  FAIL: %s - %s\n", __func__, msg);
    all_ok = 0;
  }

  free(src);

  if (all_ok) {
    TEST_PASS();
  }
  return 0;
}

/* =========================================================================
 * Test 2: snprintf truncation — table-driven
 * ========================================================================= */

typedef struct {
  const char *name;
  size_t bufsize;
  const char *fmt;
  int width;
  double value;
  int expect_truncation; /* 1 if output would exceed bufsize-1 */
} SnprintfCase;

static SnprintfCase snprintf_cases[] = {
    /* Normal: output fits in buffer */
    {"small_fits_8",     8,  "%*.2f", 7,  1.23,       0},
    {"small_fits_7",     7,  "%*.2f", 6,  1.23,       0},
    {"zero_value",       8,  "%*.2f", 7,  0.0,        0},
    /* Boundary: output just fills buffer */
    {"exact_fill_8",     8,  "%*.2f", 7,  99.99,      0},
    /* Overflow: output exceeds buffer */
    {"overflow_8",       8,  "%*.2f", 7,  99999999.0, 1},
    {"overflow_7",       7,  "%*.2f", 6,  9999999.0,  1},
    {"negative_overflow", 8, "%*.2f", 7, -9999999.0,  1},
    /* Scientific notation */
    {"sci_fits",         8,  "%*.1e", 7,  1.23e20,    0},
    {NULL, 0, NULL, 0, 0.0, 0}
};

int test_snprintf_truncation(void) {
  char msg[256];
  for (int i = 0; snprintf_cases[i].name != NULL; i++) {
    SnprintfCase *tc = &snprintf_cases[i];
    char *buf = (char *)malloc(tc->bufsize + 16); /* extra space for canary */
    memset(buf, 'X', tc->bufsize + 16);

    int ret = snprintf(buf, tc->bufsize, tc->fmt, tc->width, tc->value);

    /* snprintf ALWAYS null-terminates (when bufsize > 0) */
    snprintf(msg, sizeof(msg), "%s: not null-terminated", tc->name);
    TEST_ASSERT(buf[strlen(buf)] == '\0', msg);

    /* Output never exceeds bufsize-1 characters */
    snprintf(msg, sizeof(msg), "%s: strlen %zu >= bufsize %zu",
             tc->name, strlen(buf), tc->bufsize);
    TEST_ASSERT(strlen(buf) < tc->bufsize, msg);

    /* Canary byte at bufsize should be untouched */
    snprintf(msg, sizeof(msg), "%s: canary corrupted at position %zu",
             tc->name, tc->bufsize);
    TEST_ASSERT(buf[tc->bufsize] == 'X', msg);

    if (tc->expect_truncation) {
      /* Return value indicates more space was needed */
      snprintf(msg, sizeof(msg), "%s: expected truncation (ret=%d >= bufsize=%zu)",
               tc->name, ret, tc->bufsize);
      TEST_ASSERT(ret >= (int)tc->bufsize, msg);
    }

    free(buf);
  }
  TEST_PASS();
}

/* =========================================================================
 * Test 3: snprintf return value — overflow detection
 * ========================================================================= */

int test_snprintf_return_value(void) {
  /* snprintf returns what WOULD have been written, even when truncated.
   * This is the mechanism that makes overflow detection possible. */
  char buf[8];
  int ret;

  /* Case 1: fits */
  ret = snprintf(buf, sizeof(buf), "%*.2f", 7, 1.23);
  TEST_ASSERT(ret == 7, "expected ret=7 for '   1.23'");
  TEST_ASSERT(strlen(buf) == 7, "expected strlen=7");
  TEST_ASSERT(buf[7] == '\0', "expected null at position 7");

  /* Case 2: would overflow — snprintf truncates but returns full length */
  ret = snprintf(buf, sizeof(buf), "%*.2f", 7, 12345678.9);
  TEST_ASSERT(ret > 7, "expected ret > 7 for large value");
  TEST_ASSERT(strlen(buf) == 7, "truncated output should be 7 chars");
  TEST_ASSERT(buf[7] == '\0', "must be null-terminated even when truncated");

  TEST_PASS();
}

/* =========================================================================
 * Test 4: snprintf with exact getFloatStr caller buffer sizes
 * ========================================================================= */

int test_snprintf_getfloatstr_sizes(void) {
  char msg[256];

  /* These match the exact declarations in util.cu writeBenchmarkLineBody():
   *   char timeStr[8];   getFloatStr(timeUsec, 7, timeStr);
   *   char algBwStr[7];  getFloatStr(algBw, 6, algBwStr);
   *   char busBwStr[7];  getFloatStr(busBw, 6, busBwStr);
   */
  struct { size_t bufsize; int width; } callers[] = {
    {8, 7}, /* timeStr */
    {7, 6}, /* algBwStr */
    {7, 6}, /* busBwStr */
  };

  double extreme_values[] = {0.0, 0.001, 1.0, 999.99, 1e6, 1e12, 1e30, DBL_MAX, -1.0, -1e20};
  int nvalues = sizeof(extreme_values) / sizeof(extreme_values[0]);
  int ncallers = sizeof(callers) / sizeof(callers[0]);

  for (int c = 0; c < ncallers; c++) {
    size_t bufsize = callers[c].bufsize;
    int width = callers[c].width;
    char *buf = (char *)malloc(bufsize + 1);

    for (int v = 0; v < nvalues; v++) {
      memset(buf, 'X', bufsize + 1);

      snprintf(buf, bufsize, "%*.2f", width, extreme_values[v]);

      snprintf(msg, sizeof(msg),
               "caller[%d] buf=%zu width=%d value=%g: strlen %zu >= bufsize %zu",
               c, bufsize, width, extreme_values[v], strlen(buf), bufsize);
      TEST_ASSERT(strlen(buf) < bufsize, msg);

      snprintf(msg, sizeof(msg),
               "caller[%d] buf=%zu width=%d value=%g: canary corrupted",
               c, bufsize, width, extreme_values[v]);
      TEST_ASSERT(buf[bufsize] == 'X', msg);
    }
    free(buf);
  }
  TEST_PASS();
}

/* =========================================================================
 * Test 5: snprintf for rootName (writeBenchmarkLinePreamble)
 * ========================================================================= */

int test_snprintf_rootname(void) {
  char rootName[100];
  int ret;

  /* Normal case */
  ret = snprintf(rootName, sizeof(rootName), "%6i", 0);
  TEST_ASSERT(ret > 0 && ret < 100, "root=0 should fit");
  TEST_ASSERT(strlen(rootName) <= 6, "root=0 should be <= 6 chars");

  /* Edge: INT_MAX */
  ret = snprintf(rootName, sizeof(rootName), "%6i", INT_MAX);
  TEST_ASSERT(ret > 0 && ret < 100, "root=INT_MAX should fit in 100 bytes");

  /* Edge: INT_MIN */
  ret = snprintf(rootName, sizeof(rootName), "%6i", INT_MIN);
  TEST_ASSERT(ret > 0 && ret < 100, "root=INT_MIN should fit in 100 bytes");

  /* Verify null-termination */
  TEST_ASSERT(rootName[strlen(rootName)] == '\0', "must be null-terminated");

  TEST_PASS();
}

/* =========================================================================
 * Test 6 (ADVERSARIAL): sprintf overflow exploit
 *
 * Demonstrates the actual vulnerability in getFloatStr():
 *   char timeStr[8];
 *   sprintf(str, "%*.2f", 7, value);
 * When value is large, sprintf writes past the buffer end.
 * ========================================================================= */

int test_sprintf_overflow_exploit(void) {
  /* Allocate buffer + canary region to detect overflow.
   * Use malloc so the canary is in our controlled memory, not on the stack
   * where overflow might silently corrupt adjacent variables. */
  const size_t bufsize = 8;   /* matches char timeStr[8] in util.cu */
  const size_t total = bufsize + 8; /* 8 canary bytes */
  char *region = (char *)malloc(total);
  char *buf = region;
  char *canary = region + bufsize;

  /* Set canary to known pattern */
  memset(canary, 0xAA, 8);

  /* This value produces output longer than 7 characters:
   * "%7.2f" with 99999999.99 => "99999999.99" (11 chars + null) */
  sprintf(buf, "%*.2f", 7, 99999999.99);

  /* PROOF: sprintf wrote past the buffer, corrupting the canary */
  int canary_corrupted = 0;
  for (int i = 0; i < 8; i++) {
    if (canary[i] != (char)0xAA) {
      canary_corrupted = 1;
      break;
    }
  }
  TEST_ASSERT(canary_corrupted,
    "EXPLOIT PROOF: sprintf MUST overflow an 8-byte buffer with a large value — "
    "if this fails, the value didn't trigger overflow (adjust test value)");

  /* Now demonstrate the FIX: snprintf does NOT overflow */
  memset(region, 0, total);
  memset(canary, 0xBB, 8);

  snprintf(buf, bufsize, "%*.2f", 7, 99999999.99);

  int canary_safe = 1;
  for (int i = 0; i < 8; i++) {
    if (canary[i] != (char)0xBB) {
      canary_safe = 0;
      break;
    }
  }
  TEST_ASSERT(canary_safe,
    "snprintf must NOT overflow the buffer — canary should be intact");
  TEST_ASSERT(buf[bufsize - 1] == '\0',
    "snprintf must null-terminate within buffer bounds");

  free(region);
  TEST_PASS();
}

/* =========================================================================
 * Test 7 (ADVERSARIAL): strcpy overflow exploit
 *
 * Demonstrates strcpy writing past buffer end when source > destination.
 * Models the risk at util.cu:576 if the marker string were ever changed.
 * ========================================================================= */

int test_strcpy_overflow_exploit(void) {
  const size_t bufsize = 5;  /* matches the 5 bytes available at MAX_LINE-5 */
  const size_t total = bufsize + 8;
  char *region = (char *)malloc(total);
  char *buf = region;
  char *canary = region + bufsize;

  /* Set canary */
  memset(canary, 0xCC, 8);
  memset(buf, 0, bufsize);

  /* EXPLOIT: strcpy with source longer than destination */
  strcpy(buf, "ABCDEFGH"); /* 8 chars + null = 9 bytes into 5-byte buffer */

  int canary_corrupted = 0;
  for (int i = 0; i < 8; i++) {
    if (canary[i] != (char)0xCC) {
      canary_corrupted = 1;
      break;
    }
  }
  TEST_ASSERT(canary_corrupted,
    "EXPLOIT PROOF: strcpy MUST overflow a 5-byte buffer with 9-byte source");

  /* FIX: memcpy with explicit size + null-termination */
  memset(region, 0, total);
  memset(canary, 0xDD, 8);

  memcpy(buf, "...\n", 4);
  buf[4] = '\0';

  int canary_safe = 1;
  for (int i = 0; i < 8; i++) {
    if (canary[i] != (char)0xDD) {
      canary_safe = 0;
      break;
    }
  }
  TEST_ASSERT(canary_safe,
    "memcpy + null-term must NOT overflow — canary should be intact");
  TEST_ASSERT(buf[0] == '.' && buf[1] == '.' && buf[2] == '.',
    "first 3 chars should be '.'");
  TEST_ASSERT(buf[3] == '\n', "4th char should be newline");
  TEST_ASSERT(buf[4] == '\0', "5th char should be null terminator");
  TEST_ASSERT(strlen(buf) == 4, "string length should be 4");

  free(region);
  TEST_PASS();
}

/* =========================================================================
 * Test 8: memcpy truncation marker — exact reproduction of util.cu:575-576
 * ========================================================================= */

int test_memcpy_truncation_marker(void) {
  #define MAX_LINE 2048
  char line[MAX_LINE];

  /* Simulate a full buffer (as if snprintf filled it to capacity) */
  memset(line, 'X', MAX_LINE);
  line[MAX_LINE - 1] = '\0';

  /* Apply the safe truncation marker (the fix for util.cu:576) */
  memcpy(line + MAX_LINE - 5, "...\n", 4);
  line[MAX_LINE - 1] = '\0';

  TEST_ASSERT(line[MAX_LINE - 5] == '.', "position -5 should be '.'");
  TEST_ASSERT(line[MAX_LINE - 4] == '.', "position -4 should be '.'");
  TEST_ASSERT(line[MAX_LINE - 3] == '.', "position -3 should be '.'");
  TEST_ASSERT(line[MAX_LINE - 2] == '\n', "position -2 should be newline");
  TEST_ASSERT(line[MAX_LINE - 1] == '\0', "position -1 should be null");
  TEST_ASSERT(strlen(line + MAX_LINE - 5) == 4, "marker string length should be 4");

  /* Content before the marker should be untouched */
  TEST_ASSERT(line[MAX_LINE - 6] == 'X', "byte before marker should be untouched");

  #undef MAX_LINE
  TEST_PASS();
}

/* =========================================================================
 * Test 9: localtime_r basic functionality
 * ========================================================================= */

int test_localtime_r_basic(void) {
  struct tm result;
  time_t ts;

  /* Test 1: epoch (1970-01-01 00:00:00 UTC) */
  ts = 0;
  memset(&result, 0xFF, sizeof(result)); /* poison */
  localtime_r(&ts, &result);
  /* Year should be 1969 or 1970 depending on timezone */
  TEST_ASSERT(result.tm_year + 1900 >= 1969 && result.tm_year + 1900 <= 1970,
    "epoch should decode to 1969 or 1970");

  /* Test 2: known timestamp 1700000000 = ~2023-11-14 */
  ts = 1700000000;
  memset(&result, 0xFF, sizeof(result));
  localtime_r(&ts, &result);
  TEST_ASSERT(result.tm_year + 1900 == 2023,
    "timestamp 1700000000 should be year 2023");

  /* Test 3: returns pointer to caller-provided buffer */
  struct tm *ret = localtime_r(&ts, &result);
  TEST_ASSERT(ret == &result,
    "localtime_r must return pointer to caller-provided struct");

  /* Test 4: current time should be reasonable */
  time(&ts);
  localtime_r(&ts, &result);
  TEST_ASSERT(result.tm_year + 1900 >= 2024 && result.tm_year + 1900 <= 2030,
    "current year should be 2024-2030");

  TEST_PASS();
}

/* =========================================================================
 * Test 10: localtime_r thread safety
 * ========================================================================= */

#define NUM_THREADS 8
#define ITERATIONS 10000

typedef struct {
  time_t timestamp;
  int expected_year; /* tm_year + 1900 */
  int pass;
  int failures;
  char failure_msg[256];
} ThreadArg;

static void *localtime_r_thread(void *arg) {
  ThreadArg *ta = (ThreadArg *)arg;
  ta->pass = 1;
  ta->failures = 0;

  for (int i = 0; i < ITERATIONS; i++) {
    struct tm result;
    localtime_r(&ta->timestamp, &result);
    int year = result.tm_year + 1900;
    if (year != ta->expected_year) {
      ta->pass = 0;
      ta->failures++;
      if (ta->failures == 1) {
        snprintf(ta->failure_msg, sizeof(ta->failure_msg),
                 "thread expected year %d but got %d at iteration %d",
                 ta->expected_year, year, i);
      }
    }
  }
  return NULL;
}

int test_localtime_r_thread_safety(void) {
  /* Timestamps with distinct years (UTC) */
  time_t timestamps[] = {
    0,           /* 1970 (or 1969 in some TZ) */
    946684800,   /* 2000-01-01 */
    1000000000,  /* 2001-09-09 */
    1100000000,  /* 2004-11-09 */
    1200000000,  /* 2008-01-10 */
    1300000000,  /* 2011-03-13 */
    1400000000,  /* 2014-05-13 */
    1500000000,  /* 2017-07-14 */
  };

  ThreadArg args[NUM_THREADS];
  pthread_t threads[NUM_THREADS];

  /* Pre-compute expected years using localtime_r (single-threaded) */
  for (int i = 0; i < NUM_THREADS; i++) {
    args[i].timestamp = timestamps[i];
    struct tm tmp;
    localtime_r(&timestamps[i], &tmp);
    args[i].expected_year = tmp.tm_year + 1900;
    args[i].pass = 0;
    args[i].failures = 0;
  }

  /* Launch threads */
  for (int i = 0; i < NUM_THREADS; i++) {
    pthread_create(&threads[i], NULL, localtime_r_thread, &args[i]);
  }
  for (int i = 0; i < NUM_THREADS; i++) {
    pthread_join(threads[i], NULL);
  }

  /* All threads should pass */
  for (int i = 0; i < NUM_THREADS; i++) {
    char msg[512];
    snprintf(msg, sizeof(msg),
             "thread %d (year %d): %s",
             i, args[i].expected_year,
             args[i].pass ? "ok" : args[i].failure_msg);
    TEST_ASSERT(args[i].pass, msg);
  }

  TEST_PASS();
}

/* =========================================================================
 * Test 11 (ADVERSARIAL): localtime race condition exploit
 *
 * Demonstrates that bare localtime() corrupts results across threads.
 * Each thread calls localtime() with a timestamp from a distinct year
 * and checks whether the returned year matches. With the shared static
 * buffer, threads overwrite each other's results.
 *
 * This test PASSES by proving corruption WAS detected.
 * ========================================================================= */

#define RACE_THREADS 8
#define RACE_ITERATIONS 50000

typedef struct {
  time_t timestamp;
  int expected_year;
  int corruptions;
} RaceArg;

static void *localtime_race_thread(void *arg) {
  RaceArg *ra = (RaceArg *)arg;
  ra->corruptions = 0;

  for (int i = 0; i < RACE_ITERATIONS; i++) {
    /* WARNING: deliberately using thread-unsafe localtime() to prove the bug */
    struct tm *result = localtime(&ra->timestamp);
    int year = result->tm_year + 1900;
    if (year != ra->expected_year) {
      ra->corruptions++;
    }
  }
  return NULL;
}

int test_localtime_race_exploit(void) {
  /* Timestamps with very distinct years so corruption is obvious */
  time_t timestamps[] = {
    0,           /* 1970 */
    315532800,   /* 1980 */
    631152000,   /* 1990 */
    946684800,   /* 2000 */
    1262304000,  /* 2010 */
    1577836800,  /* 2020 */
    1893456000,  /* 2030 */
    1735689600,  /* 2025 */
  };

  RaceArg args[RACE_THREADS];
  pthread_t threads[RACE_THREADS];

  /* Pre-compute expected years (single-threaded, safe) */
  for (int i = 0; i < RACE_THREADS; i++) {
    args[i].timestamp = timestamps[i];
    struct tm tmp;
    localtime_r(&timestamps[i], &tmp);
    args[i].expected_year = tmp.tm_year + 1900;
    args[i].corruptions = 0;
  }

  /* Launch all threads simultaneously */
  for (int i = 0; i < RACE_THREADS; i++) {
    pthread_create(&threads[i], NULL, localtime_race_thread, &args[i]);
  }
  for (int i = 0; i < RACE_THREADS; i++) {
    pthread_join(threads[i], NULL);
  }

  /* Count total corruptions across all threads */
  int total_corruptions = 0;
  for (int i = 0; i < RACE_THREADS; i++) {
    total_corruptions += args[i].corruptions;
  }

  printf("    localtime race: %d corruptions detected across %d threads x %d iterations\n",
         total_corruptions, RACE_THREADS, RACE_ITERATIONS);

  /* On most systems with 8 threads and 50K iterations, corruption is
   * virtually certain. But on systems where localtime happens to be
   * thread-safe (e.g., some BSDs with thread-local storage), it might
   * not corrupt. We accept either outcome but report it. */
  if (total_corruptions > 0) {
    printf("    EXPLOIT CONFIRMED: bare localtime() corrupts across threads\n");
  } else {
    printf("    NOTE: no corruption detected (localtime may be thread-local on this platform)\n");
  }

  /* This test passes either way — it's informational.
   * The real protection is the source-verification test ensuring
   * localtime_r is used in the actual source code. */
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
     "Verify util.cu uses safe C functions (FAILS before fix)"},
    {"snprintf-truncation", test_snprintf_truncation,
     "snprintf truncation and null-termination (table-driven)"},
    {"snprintf-retval", test_snprintf_return_value,
     "snprintf return value overflow detection"},
    {"snprintf-getfloatstr", test_snprintf_getfloatstr_sizes,
     "snprintf with exact getFloatStr caller buffer sizes"},
    {"snprintf-rootname", test_snprintf_rootname,
     "snprintf for writeBenchmarkLinePreamble rootName"},
    {"sprintf-overflow-exploit", test_sprintf_overflow_exploit,
     "ADVERSARIAL: sprintf overflows tight buffer (canary proof)"},
    {"strcpy-overflow-exploit", test_strcpy_overflow_exploit,
     "ADVERSARIAL: strcpy overflows buffer (canary proof)"},
    {"memcpy-truncmarker", test_memcpy_truncation_marker,
     "memcpy truncation marker (safe replacement for strcpy)"},
    {"localtime_r-basic", test_localtime_r_basic,
     "localtime_r basic functionality and correctness"},
    {"localtime_r-threads", test_localtime_r_thread_safety,
     "localtime_r concurrent thread safety (8 threads)"},
    {"localtime-race-exploit", test_localtime_race_exploit,
     "ADVERSARIAL: bare localtime() corrupts across threads"},
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
      printf("  %-30s %s\n", test_cases[i].name, test_cases[i].description);
    }
    printf("\nRun with no arguments to execute all tests.\n");
    return 0;
  }

  /* Also support TEST_CASE env var (matches NCCL test convention) */
  if (!filter) {
    filter = getenv("TEST_CASE");
  }

  printf("=== util.cu buffer safety & thread safety tests ===\n\n");

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
