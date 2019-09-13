#include <iostream>
#include "omp.h"
uint64_t N=4294967296ULL, sum;
int main(int argc, const char * argv[]) {
#pragma omp parallel for reduction(+:sum)
  for (uint64_t i=0; i<N; i++)
    sum += i;
  std::cout << sum << std::endl;
  return 0;
}
