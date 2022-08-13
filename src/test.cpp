// Type your code here, or load an example.
#include <algorithm>
#include <random>
#include <vector>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <immintrin.h>
#include <chrono>
#include <any>

template<typename... Vals>
bool allEqual(Vals... vals) {
    std::vector<std::any> input({vals...});
    if (std::all_of(input.begin(), input.end(), [&] (std::any i) {return std::any_cast<int>(i) == std::any_cast<int>(input[0]);})){
        return true;
    } else {
        for (auto&& val : input) {
            std::cerr << std::any_cast<int>(val) << " ";
        }
        std::cerr << std::endl;
        return false;
    }
    return false;
}

int getSize() {
    srand (time(NULL));
    return rand() % 2000; // Assume size of output layer is about this at most
}

void populateVec(float * logits, size_t size) {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> d{-5,5};
    for (size_t i = 0; i<size; i++) {
        logits[i] = d(gen);
    }
}

int max_elem(float * vec, size_t size) {
    auto elem = std::max_element(vec, vec + size);
    return std::distance(vec, elem);
}

int max_elem_manual(float * vec, size_t size) {
    float maxVal = vec[0]; // Ignore empty vec case
    int max_idx = 0;
    for (size_t i = 0; i < size; i++) {
        if (maxVal < vec[i]) {
            max_idx = i;
            maxVal = vec[i];
        }
    }
    return max_idx;
}
#ifdef __AVX512F__
int max_elem_avx512(float * vec, size_t size) {
    float maxVal = vec[0];
    int max_idx = 0;
    div_t setup = div(size, 32);
    int overhang = setup.rem;
    int seq = setup.quot;
    for (int i = 0; i < seq*32; i+=32) {
        __m512 first = _mm512_load_ps(&vec[i]);
        __m512 second = _mm512_load_ps(&vec[i+16]);
        __m512 maxd = _mm512_max_ps(first, second);
        // Max reduce is not an intrinsic. it gets converted to a sequence of instructions
        float max_single = _mm512_reduce_max_ps(maxd);
        if (max_single > maxVal) {
            for (int j = 0; j<32; j++) {
                if (vec[i+j] == max_single) {
                    maxVal = vec[i+j];
                    max_idx = i+j;
                    break;
                }
            }
        }
    }

    for (int i = seq*32; i < seq*32 + overhang; i++) {
        if (maxVal < vec[i]) {
            max_idx = i;
            maxVal = vec[i];
        }
    }
    return max_idx;
}

int max_elem_avx512_2(float * vec, size_t size) {
    float maxVal = vec[0];
    int max_idx = 0;
    div_t setup = div(size, 16);
    int overhang = setup.rem;
    int seq = setup.quot;
    for (int i = 0; i < seq*16; i+=16) {
        float max_single = _mm512_reduce_max_ps(_mm512_load_ps(&vec[i]));
        if (max_single > maxVal) {
            for (int j = 0; j<16; j++) {
                if (vec[i+j] == max_single) {
                    maxVal = vec[i+j];
                    max_idx = i+j;
                    break;
                }
            }
        }
    }

    for (int i = seq*16; i < seq*16 + overhang; i++) {
        if (maxVal < vec[i]) {
            max_idx = i;
            maxVal = vec[i];
        }
    }
    return max_idx;
}

int max_elem_avx512_3(float * vec, size_t size) {
    float maxVal = vec[0];
    int max_idx = 0;
    div_t setup = div(size, 16);
    int overhang = setup.rem;
    int seq = setup.quot;
    __m512 maxvalVec = _mm512_set1_ps(maxVal);
    for (int i = 0; i < seq*16; i+=16) {
        auto res = _mm512_cmp_ps_mask(maxvalVec, _mm512_load_ps(&vec[i]), _CMP_LT_OS);
        // We might have more than one increased matches, so not sure if this can be further optimised
        if (res != 0) {
            for (int j = 0; j<16; j++) {
                if (vec[i+j] > maxVal) {
                    maxVal = vec[i+j];
                    max_idx = i+j;
                }
            }
            maxvalVec = _mm512_set1_ps(maxVal);
        }
    }

    for (int i = seq*16; i < seq*16 + overhang; i++) {
        if (maxVal < vec[i]) {
            max_idx = i;
            maxVal = vec[i];
        }
    }
    return max_idx;
}
#endif

// MAX_REDUCE is not available before avx512
#ifdef __AVX__
int max_elem_avx_3(float * vec, size_t size) {
    float maxVal = vec[0];
    int max_idx = 0;
    div_t setup = div(size, 8);
    int overhang = setup.rem;
    int seq = setup.quot;
    __m256 maxvalVec = _mm256_set1_ps(maxVal);
    for (int i = 0; i < seq*8; i+=8) {
        __m256 res = _mm256_cmp_ps(maxvalVec, _mm256_load_ps(&vec[i]), _CMP_LT_OS);
        if (_mm256_movemask_ps(res) != 0) {
            for (int j = 0; j<8; j++) {
                if (vec[i+j] > maxVal) {
                    maxVal = vec[i+j];
                    max_idx = i+j;
                }
            }
            maxvalVec = _mm256_set1_ps(maxVal);
        }
    }

    for (int i = seq*8; i < seq*8 + overhang; i++) {
        if (maxVal < vec[i]) {
            max_idx = i;
            maxVal = vec[i];
        }
    }
    return max_idx;
}
#endif

#ifdef __SSE4_1__
int max_elem_sse_3(float * vec, size_t size) {
    float maxVal = vec[0];
    int max_idx = 0;
    div_t setup = div(size, 4);
    int overhang = setup.rem;
    int seq = setup.quot;
    __m128 maxvalVec = _mm_set1_ps(maxVal);
    for (int i = 0; i < seq*4; i+=4) {
        __m128 res = _mm_cmplt_ps(maxvalVec, _mm_load_ps(&vec[i]));
        // We might have more than one increased matches, so not sure if this can be further optimised
        if (_mm_movemask_ps(res) != 0) {
            for (int j = 0; j<4; j++) {
                if (vec[i+j] > maxVal) {
                    maxVal = vec[i+j];
                    max_idx = i+j;
                }
            }
            maxvalVec = _mm_set1_ps(maxVal);
        }
    }

    for (int i = seq*4; i < seq*4 + overhang; i++) {
        if (maxVal < vec[i]) {
            max_idx = i;
            maxVal = vec[i];
        }
    }
    return max_idx;
}
#endif


int main() {
    std::chrono::duration<double> elapsed_seconds(0);
    std::chrono::duration<double> elapsed_seconds_seq(0);
    std::chrono::duration<double> elapsed_seconds_avx512(0);
    std::chrono::duration<double> elapsed_seconds_avx512_2(0);
    std::chrono::duration<double> elapsed_seconds_avx512_3(0);
    std::chrono::duration<double> elapsed_seconds_avx_3(0);
    std::chrono::duration<double> elapsed_seconds_sse_3(0);
    for (int i = 0; i < 1000000; i++) {
        size_t size = getSize();
        float *logits = (float *)aligned_alloc(1024, size*sizeof(float));
        populateVec(logits, size);

        // Baselines
        auto start = std::chrono::steady_clock::now();
        int mymax = max_elem(logits, size);
        auto end = std::chrono::steady_clock::now();

        // Mine:
        auto start2 = std::chrono::steady_clock::now();
        int mymax2 = max_elem_manual(logits, size);
        auto end2 = std::chrono::steady_clock::now();

        elapsed_seconds += end-start;
        elapsed_seconds_seq += end2-start2;

        // avx512, first attempt
#ifdef __AVX512F__
        auto start3 = std::chrono::steady_clock::now();
        int mymax3 = max_elem_avx512(logits, size);
        auto end3 = std::chrono::steady_clock::now();

        // avx512, second attempt
        auto start4 = std::chrono::steady_clock::now();
        int mymax4 = max_elem_avx512_2(logits, size);
        auto end4 = std::chrono::steady_clock::now();

        // avx512, third attempt
        auto start5 = std::chrono::steady_clock::now();
        int mymax5 = max_elem_avx512_3(logits, size);
        auto end5 = std::chrono::steady_clock::now();

        // Check for correctness
        if (!allEqual(mymax, mymax2, mymax3, mymax4, mymax5)) {
            std::cerr << "Mymax1: " << logits[mymax] << " MyMax2 " << logits[mymax2]  << 
            " MyMax3 " << logits[mymax3] << " MyMax4 " << logits[mymax4] 
            << " MyMax5 " << logits[mymax5] << " size: " << size << std::endl;
            break;
        }
        elapsed_seconds_avx512 += end3-start3;
        elapsed_seconds_avx512_2 += end4-start4;
        elapsed_seconds_avx512_3 += end5-start5;
#endif

#ifdef __AVX__
        auto start6 = std::chrono::steady_clock::now();
        int mymax6 = max_elem_avx_3(logits, size);
        auto end6 = std::chrono::steady_clock::now();

        // Check for correctness
        if (!allEqual(mymax, mymax6)) {
            std::cerr << "Mymax1: " << logits[mymax] << " MyMaxAVX " << logits[mymax6]  <<  " size: " << size << std::endl;
            break;
        }
        elapsed_seconds_avx_3 += end6-start6;
#endif

#ifdef __SSE4_1__
        auto start7 = std::chrono::steady_clock::now();
        int mymax7 = max_elem_sse_3(logits, size);
        auto end7 = std::chrono::steady_clock::now();

        // Check for correctness
        if (!allEqual(mymax, mymax7)) {
            std::cerr << "Mymax1: " << logits[mymax] << " MyMaxSSE " << logits[mymax7]  <<  " size: " << size << std::endl;
            break;
        }
        elapsed_seconds_sse_3 += end7-start7;
#endif
        free(logits);
    }
    std::cout << "Elapsed time. Baselines: \n"
    << "std::max_element: "<< elapsed_seconds.count() << "s\n"
    << "simple_seq: " << elapsed_seconds_seq.count() << "s\n"
    << "AVX512F:\n"
#ifdef __AVX512F__
    << "max_element_avx512: "<< elapsed_seconds_avx512.count() << "s\n"
    << "max_element_avx512_2: "<< elapsed_seconds_avx512_2.count() << "s\n"
    << "max_element_avx512_3: "<< elapsed_seconds_avx512_3.count() << "s\n"
#endif
#ifdef __AVX__
    << "AVX:\n"
    << "max_element_avx_3: "<< elapsed_seconds_avx_3.count() << "s\n"
#endif
#ifdef __SSE4_2__
    << "SSE:\n"
    << "max_element_sse_3: "<< elapsed_seconds_sse_3.count() << "s\n";
#else
;
#endif
    return 0;
}
