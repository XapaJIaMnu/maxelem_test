# maxelem_test
Tests different implementations of `std::max_element` for a specific case of output layer shortlisting. The algorithms I have use perform SIMD `max`/`max_reduce`/`compare` and only finding the index of a max_element in case a new one has been found in the SIMD operation. This algorithm will perform very poorly if the array is sorted, or almost sorted, but quite good in all other cases. Results seem to vary by compiler:

## Compile
```bash
mkdir build
cd build
cmake ..
make
./test.out
```

## Tests
Tested on different hardware with different compilers. 100000 iterations of float array. The float array is of size randomly generated up to 2000 elements, and the floats inside are drawn from the uniform distribution \[5:5\].
### Cooper lake results

|                                | GCC 11.2 | clang 14 | icc 2022.1.0 |
|--------------------------------|----------|----------|--------------|
| std::max_element               | 2.6696s  | 0.4221s  | 0.4662s      |
| sequential                     | 1.0831s  | 1.1924s  | 1.1472s      |
| AVX512 max + max_reduce        | 0.2412s  | 0.2152s  | 0.2142s      |
| AVX512 max_reduce only         | 0.2570s  | 0.2629s  | 0.2325s      |
| AVX512 cmp_ps_mask             | 0.1884s  | 0.1826s  | 0.1833s      |
| AVX512 ^ + vectorized overhang | 0.2097s  | 0.2089s  | 0.2072s      |
| AVX cmp_ps + movemask          | 0.2181s  | 0.1697s  | 0.1702s      |
| SSE cmplt_psp + movemask       | 0.2692s  | 0.2051s  | 0.2221s      |

### Ryzen 9 5900HS results

|                          | GCC 11.2 | clang 14 |
|--------------------------|----------|----------|
| std::max_element         | 2.4101s  | 0.7221s  |
| sequential               | 0.3252s  | 0.3518s  |
| AVX cmp_ps + movemask    | 0.1476s  | 0.1214s  |
| SSE cmplt_psp + movemask | 0.1693s  | 0.1468s  |

## Conclusions
- `std::max_element` is really slow, especially when compiled with `gcc`, where it is slower than even trivial sequential implementation.
- On AVX512 systems, `clang` and `icc` manage to do better job at optimising `std::max_element`, but hand written assembly implementations are faster.
- Unsurprisingly SIMD `max_reduce` implementations are slow, as horizontal reductions are in general slow on x86.
- Using `_mm512_cmp_ps_mask` or equivalent instructions (based on the available ISA) seem to be the winning solution.
- Having vectorised overhang, where we drop to using `__m256` and `__m128` before falling back to sequential code are slower than just using sequential code. CPU state switching between AVX512/AVX/SSE mode to blame? CPU frequency drops to blame?
- `clang` and `icc` produce better results with the AVX implementation based on `_mm256_cmp_ps` and `_mm256_movemask_ps`. These are also the fastest times produced.

## Future work
- Inspect assembly to find out why vectorised overhang is slower.
- Inspect assembly to find out why AVX is faster with ICC/GCC.
