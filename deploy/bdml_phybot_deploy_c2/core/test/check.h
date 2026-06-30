#pragma once
#include <cstdio>
inline int g_check_failures = 0;
#define CHECK(cond)                                                     \
  do {                                                                  \
    if (!(cond)) {                                                      \
      ++g_check_failures;                                               \
      std::printf("FAIL: %s (%s:%d)\n", #cond, __FILE__, __LINE__);     \
    }                                                                   \
  } while (0)
