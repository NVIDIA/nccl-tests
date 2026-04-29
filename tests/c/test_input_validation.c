/*************************************************************************
 * Unit tests for input validation and type conversion safety
 *
 * Tests verify that:
 *   1. atoi/atof silently fail on garbage and overflow (adversarial)
 *   2. strtol/strtod detect garbage, overflow, and partial parse
 *   3. The exact NCCL_TESTS_DEVICE parsing pattern works safely
 *   4. Signed-to-unsigned and uint64-to-double casts are explicit
 *   5. Source files have been patched to use safe alternatives
 *
 * The source verification test (test_source_verified) intentionally FAILS
 * before the fix is applied, demonstrating the bugs exist in unfixed code.
 *
 * Compile: gcc -Wall -Wextra -Wno-format-truncation -g -std=c99 -D_GNU_SOURCE \
 *          -o test_input_validation test_input_validation.c -lpthread
 * Run:     ./test_input_validation
 *************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>
#include <errno.h>
#include <float.h>
#include <math.h>

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

/* =========================================================================
 * Test 1: Source verification
 * INTENTIONALLY FAILS before the fix is applied.
 * ========================================================================= */

int test_source_verified(void) {
  int all_ok = 1;
  char msg[512];

  /* Check 1: common.cu should not use atoi( */
  const char *common_path = "../../src/common.cu";
  char *common_src = read_file(common_path);
  if (!common_src) {
    snprintf(msg, sizeof(msg), "Cannot read %s (run from tests/c/)", common_path);
    printf("  SKIP: %s - %s\n", __func__, msg);
  } else {
    if (strstr(common_src, "atoi(") != NULL) {
      snprintf(msg, sizeof(msg),
               "%s: still uses atoi() — expected strtol()",
               common_path);
      printf("  FAIL: %s - %s\n", __func__, msg);
      all_ok = 0;
    }
    if (strstr(common_src, "atof(") != NULL) {
      snprintf(msg, sizeof(msg),
               "%s: still uses atof() — expected strtod()",
               common_path);
      printf("  FAIL: %s - %s\n", __func__, msg);
      all_ok = 0;
    }
    free(common_src);
  }

  /* Check 2: util.cu should not use atoi( */
  const char *util_path = "../../src/util.cu";
  char *util_src = read_file(util_path);
  if (!util_src) {
    snprintf(msg, sizeof(msg), "Cannot read %s (run from tests/c/)", util_path);
    printf("  SKIP: %s - %s\n", __func__, msg);
  } else {
    if (strstr(util_src, "atoi(") != NULL) {
      snprintf(msg, sizeof(msg),
               "%s: still uses atoi() — expected strtol()",
               util_path);
      printf("  FAIL: %s - %s\n", __func__, msg);
      all_ok = 0;
    }
    free(util_src);
  }

  /* Check 3: timer.cc should use static_cast */
  const char *timer_path = "../../src/timer.cc";
  char *timer_src = read_file(timer_path);
  if (!timer_src) {
    snprintf(msg, sizeof(msg), "Cannot read %s (run from tests/c/)", timer_path);
    printf("  SKIP: %s - %s\n", __func__, msg);
  } else {
    if (strstr(timer_src, "static_cast") == NULL) {
      snprintf(msg, sizeof(msg),
               "%s: no static_cast found — implicit narrowing conversions",
               timer_path);
      printf("  FAIL: %s - %s\n", __func__, msg);
      all_ok = 0;
    }
    free(timer_src);
  }

  if (all_ok) {
    TEST_PASS();
  }
  return 0;
}

/* =========================================================================
 * Test 2 (ADVERSARIAL): atoi garbage input exploit
 *
 * Demonstrates that atoi("abc") silently returns 0, which is
 * indistinguishable from atoi("0"). For NCCL_TESTS_DEVICE=abc,
 * the current code silently selects GPU 0 instead of erroring.
 * ========================================================================= */

int test_atoi_garbage_exploit(void) {
  /* atoi on garbage returns 0 — same as valid "0" */
  int result_garbage = atoi("abc");
  int result_zero = atoi("0");

  TEST_ASSERT(result_garbage == 0,
    "atoi('abc') should return 0 (no error detection)");
  TEST_ASSERT(result_zero == 0,
    "atoi('0') should return 0 (valid input)");
  TEST_ASSERT(result_garbage == result_zero,
    "EXPLOIT PROOF: atoi cannot distinguish 'abc' from '0' — "
    "both return 0, so NCCL_TESTS_DEVICE=abc silently selects GPU 0");

  /* atoi on empty string also returns 0 */
  int result_empty = atoi("");
  TEST_ASSERT(result_empty == 0,
    "atoi('') returns 0 — indistinguishable from valid '0'");

  /* atoi on mixed input takes leading digits, ignores rest */
  int result_mixed = atoi("42xyz");
  TEST_ASSERT(result_mixed == 42,
    "atoi('42xyz') returns 42 — silently ignores trailing garbage");

  /* Now demonstrate the FIX: strtol detects all these cases */
  char *end;

  /* garbage: no digits consumed */
  errno = 0;
  strtol("abc", &end, 10);
  TEST_ASSERT(end != NULL && *end == 'a',
    "strtol('abc') leaves endptr at 'a' — detects no valid digits");

  /* empty: no digits consumed */
  errno = 0;
  const char *empty = "";
  strtol(empty, &end, 10);
  TEST_ASSERT(end == empty,
    "strtol('') leaves endptr at start — detects empty input");

  /* mixed: endptr points to trailing garbage */
  errno = 0;
  strtol("42xyz", &end, 10);
  TEST_ASSERT(end != NULL && *end == 'x',
    "strtol('42xyz') leaves endptr at 'x' — detects trailing garbage");

  TEST_PASS();
}

/* =========================================================================
 * Test 3 (ADVERSARIAL): atoi integer overflow exploit
 *
 * Demonstrates that atoi() has undefined behavior on overflow.
 * On most platforms it wraps silently. strtol detects via ERANGE.
 * ========================================================================= */

int test_atoi_overflow_exploit(void) {
  /* A value that overflows int (assuming 32-bit int) but may fit in long.
   * Use a value that is clearly > INT_MAX (2147483647). */
  const char *huge_for_int = "99999999999";

  /* atoi on int overflow: undefined behavior, typically wraps */
  int result = atoi(huge_for_int);
  printf("    atoi(\"%s\") = %d (undefined behavior, typically wraps)\n",
         huge_for_int, result);
  /* atoi returns an int, which cannot hold 99999999999.
   * The result is undefined behavior — on most platforms it wraps.
   * We verify it didn't somehow produce a "reasonable" GPU ID. */
  TEST_ASSERT(result < 0 || result > 1000,
    "EXPLOIT PROOF: atoi wraps a value > INT_MAX to a garbage int — "
    "this would select a nonsensical GPU ID");

  /* strtol parses it correctly into a long, then we can range-check */
  char *end;
  errno = 0;
  long val = strtol(huge_for_int, &end, 10);
  TEST_ASSERT(errno != ERANGE && *end == '\0',
    "strtol parses 99999999999 correctly into a long");
  TEST_ASSERT(val > INT_MAX,
    "strtol detects value > INT_MAX — caller can reject for int usage");

  /* For LONG overflow, use a truly huge number */
  const char *huge_for_long = "999999999999999999999999999999";
  errno = 0;
  val = strtol(huge_for_long, &end, 10);
  TEST_ASSERT(errno == ERANGE,
    "strtol detects LONG overflow via ERANGE");
  TEST_ASSERT(val == LONG_MAX,
    "strtol returns LONG_MAX on positive overflow");

  /* Negative LONG overflow */
  const char *neg_huge = "-999999999999999999999999999999";
  errno = 0;
  val = strtol(neg_huge, &end, 10);
  TEST_ASSERT(errno == ERANGE,
    "strtol detects negative LONG overflow via ERANGE");
  TEST_ASSERT(val == LONG_MIN,
    "strtol returns LONG_MIN on negative overflow");

  TEST_PASS();
}

/* =========================================================================
 * Test 4 (ADVERSARIAL): atof garbage input exploit
 *
 * Demonstrates that atof("xyz") silently returns 0.0, which is
 * indistinguishable from atof("0"). For NCCL_TESTS_MIN_BW=garbage,
 * the current code silently uses 0.0 as the bandwidth threshold.
 * ========================================================================= */

int test_atof_garbage_exploit(void) {
  /* atof on garbage returns 0.0 — same as valid "0" */
  double result_garbage = atof("not_a_number");
  double result_zero = atof("0");

  TEST_ASSERT(result_garbage == 0.0,
    "atof('not_a_number') should return 0.0 (no error detection)");
  TEST_ASSERT(result_zero == 0.0,
    "atof('0') should return 0.0 (valid input)");
  TEST_ASSERT(result_garbage == result_zero,
    "EXPLOIT PROOF: atof cannot distinguish 'not_a_number' from '0' — "
    "both return 0.0, so NCCL_TESTS_MIN_BW=garbage silently sets threshold to 0");

  /* Now demonstrate the FIX: strtod detects garbage */
  char *end;
  errno = 0;
  double val = strtod("not_a_number", &end);
  (void)val;
  TEST_ASSERT(end != NULL && strcmp(end, "not_a_number") == 0,
    "strtod('not_a_number') leaves endptr unchanged — detects garbage input");

  /* strtod correctly parses valid input */
  errno = 0;
  val = strtod("3.14", &end);
  TEST_ASSERT(fabs(val - 3.14) < 1e-10,
    "strtod('3.14') correctly parses to ~3.14");
  TEST_ASSERT(*end == '\0',
    "strtod('3.14') consumes entire string");

  TEST_PASS();
}

/* =========================================================================
 * Test 5: strtol validation — table-driven
 * ========================================================================= */

typedef struct {
  const char *name;
  const char *input;
  int expect_valid;
  long expect_value;  /* only checked if expect_valid */
} StrtolCase;

static StrtolCase strtol_cases[] = {
    {"zero",       "0",            1, 0},
    {"positive",   "42",           1, 42},
    {"negative",   "-7",           1, -7},
    {"leading_ws", "  123",        1, 123},
    {"hex",        "0xff",         1, 255},
    {"garbage",    "abc",          0, 0},
    {"empty",      "",             0, 0},
    {"mixed",      "42xyz",        0, 0},  /* trailing garbage = invalid */
    {"overflow",   "999999999999999999999999999999", 0, 0}, /* overflows long */
    {"just_sign",  "-",            0, 0},
    {NULL, NULL, 0, 0}
};

int test_strtol_validation(void) {
  char msg[256];
  for (int i = 0; strtol_cases[i].name != NULL; i++) {
    StrtolCase *tc = &strtol_cases[i];
    char *end;
    errno = 0;
    long val = strtol(tc->input, &end, 0);
    int valid = (end != tc->input && *end == '\0' && errno != ERANGE);

    snprintf(msg, sizeof(msg), "%s: expected valid=%d, got valid=%d",
             tc->name, tc->expect_valid, valid);
    TEST_ASSERT(valid == tc->expect_valid, msg);

    if (tc->expect_valid) {
      snprintf(msg, sizeof(msg), "%s: expected value=%ld, got %ld",
               tc->name, tc->expect_value, val);
      TEST_ASSERT(val == tc->expect_value, msg);
    }
  }
  TEST_PASS();
}

/* =========================================================================
 * Test 6: strtod validation — table-driven
 * ========================================================================= */

typedef struct {
  const char *name;
  const char *input;
  int expect_valid;
  double expect_value;
} StrtodCase;

static StrtodCase strtod_cases[] = {
    {"zero",      "0",       1, 0.0},
    {"pi",        "3.14159", 1, 3.14159},
    {"negative",  "-2.5",    1, -2.5},
    {"sci",       "1.5e3",   1, 1500.0},
    {"garbage",   "xyz",     0, 0.0},
    {"empty",     "",        0, 0.0},
    {"mixed",     "3.14abc", 0, 0.0},
    {NULL, NULL, 0, 0.0}
};

int test_strtod_validation(void) {
  char msg[256];
  for (int i = 0; strtod_cases[i].name != NULL; i++) {
    StrtodCase *tc = &strtod_cases[i];
    char *end;
    errno = 0;
    double val = strtod(tc->input, &end);
    int valid = (end != tc->input && *end == '\0' && errno != ERANGE);

    snprintf(msg, sizeof(msg), "%s: expected valid=%d, got valid=%d",
             tc->name, tc->expect_valid, valid);
    TEST_ASSERT(valid == tc->expect_valid, msg);

    if (tc->expect_valid) {
      snprintf(msg, sizeof(msg), "%s: expected value=%g, got %g",
               tc->name, tc->expect_value, val);
      TEST_ASSERT(fabs(val - tc->expect_value) < 1e-6, msg);
    }
  }
  TEST_PASS();
}

/* =========================================================================
 * Test 7: Reproduces the exact NCCL_TESTS_DEVICE pattern with strtol
 * ========================================================================= */

int test_strtol_device_id_pattern(void) {
  /* This is the safe replacement for:
   *   char* envstr = getenv("NCCL_TESTS_DEVICE");
   *   int gpu0 = envstr ? atoi(envstr) : -1;
   */

  /* Simulate various env var values */
  const char *test_values[] = {"0", "1", "7", "-1", "abc", "", "99999999999", "3 ", NULL};
  int expected_gpu[]        = { 0,   1,   7,   -1,   -1,  -1,      -1,         -1};

  for (int i = 0; test_values[i] != NULL; i++) {
    const char *envstr = test_values[i];
    int gpu0 = -1;

    /* Safe parsing pattern */
    char *end;
    errno = 0;
    long val = strtol(envstr, &end, 10);
    if (end != envstr && *end == '\0' && errno != ERANGE &&
        val >= INT_MIN && val <= INT_MAX) {
      gpu0 = (int)val;
    }

    char msg[256];
    snprintf(msg, sizeof(msg),
             "input='%s': expected gpu0=%d, got %d",
             envstr, expected_gpu[i], gpu0);
    TEST_ASSERT(gpu0 == expected_gpu[i], msg);
  }

  TEST_PASS();
}

/* =========================================================================
 * Test 8: Signed-to-unsigned cast demonstration
 *
 * Shows that a negative signed value becomes a large unsigned value
 * without an explicit cast. The cast makes this conversion visible
 * and intentional rather than implicit and surprising.
 * ========================================================================= */

int test_signed_to_unsigned_cast(void) {
  /* Simulates timer.cc line 10: chrono count() returns signed long,
   * assigned to uint64_t */
  long signed_val = 1000000000L; /* 1 second in nanoseconds */
  uint64_t unsigned_val = (uint64_t)signed_val; /* explicit cast */
  TEST_ASSERT(unsigned_val == 1000000000ULL,
    "positive signed->unsigned preserves value");

  /* Negative value: explicit cast makes the wrap-around intentional */
  long neg_val = -1L;
  uint64_t neg_unsigned = (uint64_t)neg_val;
  TEST_ASSERT(neg_unsigned == UINT64_MAX,
    "negative signed->unsigned wraps to UINT64_MAX (explicit cast documents intent)");

  /* In timer.cc, durations are always non-negative, so the conversion
   * is safe. The static_cast just silences the compiler warning and
   * documents that the developer considered the sign change. */

  TEST_PASS();
}

/* =========================================================================
 * Test 9: uint64_t-to-double precision loss demonstration
 *
 * Shows that large uint64_t values lose precision when converted to
 * double (which has only 53 bits of mantissa). For timer.cc this is
 * fine (nanosecond counts don't exceed 2^53), but the cast makes it
 * explicit.
 * ========================================================================= */

int test_uint64_to_double_precision(void) {
  /* Small values: no precision loss */
  uint64_t small = 1000000000ULL; /* 1e9 — fits in double exactly */
  double d_small = (double)small;
  TEST_ASSERT(d_small == 1000000000.0,
    "small uint64_t converts to double without precision loss");

  /* Large values: precision loss occurs */
  uint64_t large = (1ULL << 53) + 1; /* 2^53 + 1 — exceeds double mantissa */
  double d_large = (double)large;
  uint64_t roundtrip = (uint64_t)d_large;

  printf("    uint64 = %lu, double = %.0f, roundtrip = %lu\n",
         large, d_large, roundtrip);
  TEST_ASSERT(roundtrip != large,
    "PRECISION PROOF: uint64_t > 2^53 loses precision when cast to double — "
    "explicit cast documents this is intentional");

  /* For timer.cc: nanosecond elapsed time.
   * 2^53 nanoseconds = ~104 days.
   * A performance benchmark never runs that long,
   * so the conversion is safe in practice. */
  uint64_t realistic_ns = 5000000000ULL; /* 5 seconds in ns */
  double realistic_sec = 1.e-9 * (double)realistic_ns;
  TEST_ASSERT(fabs(realistic_sec - 5.0) < 1e-9,
    "realistic timer value converts precisely");

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
     "Verify source uses strtol/strtod/static_cast (FAILS before fix)"},
    {"atoi-garbage-exploit", test_atoi_garbage_exploit,
     "ADVERSARIAL: atoi('abc') indistinguishable from atoi('0')"},
    {"atoi-overflow-exploit", test_atoi_overflow_exploit,
     "ADVERSARIAL: atoi('99999999999') wraps/UB vs strtol ERANGE"},
    {"atof-garbage-exploit", test_atof_garbage_exploit,
     "ADVERSARIAL: atof('xyz') indistinguishable from atof('0')"},
    {"strtol-validation", test_strtol_validation,
     "strtol table-driven: valid ints, garbage, overflow"},
    {"strtod-validation", test_strtod_validation,
     "strtod table-driven: valid doubles, garbage"},
    {"strtol-device-id", test_strtol_device_id_pattern,
     "Exact NCCL_TESTS_DEVICE safe parsing pattern"},
    {"signed-unsigned-cast", test_signed_to_unsigned_cast,
     "Explicit signed-to-unsigned cast demonstration"},
    {"uint64-double-precision", test_uint64_to_double_precision,
     "uint64_t-to-double precision loss proof"},
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

  if (!filter) {
    filter = getenv("TEST_CASE");
  }

  printf("=== input validation & type conversion tests ===\n\n");

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
