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

#define MAX_OUTPUT_LAYER_SIZE 2000
#define ITERATIONS 1000000

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
    return rand() % MAX_OUTPUT_LAYER_SIZE; // Assume size of output layer is about this at most
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

// Templated implementaiton of 3 + overhang

template <class Register> static inline Register set1_ps(float& to);
template <class Register> static inline Register load_ps(float const* from);
template <class Register, class Mask> inline Mask cmplt_ps_mask(Register& maxVal, Register compareTRG);

template <> inline __m128 set1_ps<__m128>(float& to) {
  return _mm_set1_ps(to);
}

template <> inline __m256 set1_ps<__m256>(float& to) {
  return _mm256_set1_ps(to);
}

template <> inline __m512 set1_ps<__m512>(float& to) {
  return _mm512_set1_ps(to);
}

// load_ps
template <> inline __m128 load_ps<__m128>(const float* from) {
  return _mm_load_ps(from);
}

template <> inline __m256 load_ps<__m256>(const float* from) {
  return _mm256_load_ps(from);
}

template <> inline __m512 load_ps<__m512>(const float* from) {
  return _mm512_load_ps(from);
}
// Mask
template <> inline __mmask16 cmplt_ps_mask(__m512& maxVal, __m512 compareTRG) {
    return _mm512_cmp_ps_mask(maxVal, compareTRG, _CMP_LT_OS);
}

template <> inline __mmask8 cmplt_ps_mask(__m256& maxVal, __m256 compareTRG) {
    return _mm256_cmp_ps_mask(maxVal, compareTRG, _CMP_LT_OS);
}

template <> inline __mmask8 cmplt_ps_mask(__m128& maxVal, __m128 compareTRG) {
    return _mm_cmp_ps_mask(maxVal, compareTRG, _CMP_LT_OS);
}

template<class Register, class Mask> 
inline std::tuple<int, float> max_elem_avx512_3_recur(float * vec, size_t size, size_t start, float curMaxVal, float cur_max_idx) {
    constexpr int step_size = (sizeof(Register)/sizeof(float));
    float maxVal = curMaxVal;
    int max_idx = cur_max_idx;
    div_t setup = div(size, step_size);
    int overhang = setup.rem;
    int seq = setup.quot;
    Register maxvalVec = set1_ps<Register>(maxVal);
    for (int i = 0; i < seq*step_size; i+=step_size) {
        auto res = cmplt_ps_mask<Register, Mask>(maxvalVec, load_ps<Register>(&vec[i]));
        // We might have more than one increased matches, so not sure if this can be further optimised
        if (res != 0) {
            for (int j = 0; j<step_size; j++) {
                if (vec[i+j] > maxVal) {
                    maxVal = vec[i+j];
                    max_idx = i+j + start;
                }
            }
            maxvalVec = set1_ps<Register>(maxVal);
        }
    } 
    // AVX 512case -> Do the overhang in __m256
    if constexpr(step_size == 16) {
        return max_elem_avx512_3_recur<__m256, __mmask8>(vec + seq*step_size, overhang, start + seq*step_size, maxVal, max_idx);
    } else if constexpr(step_size == 8) { // Do the overhang in __m128
        return max_elem_avx512_3_recur<__m128, __mmask8>(vec + seq*step_size, overhang, start + seq*step_size, maxVal, max_idx);
    } else { // Do the overhang sequentially
        for (int i = seq*step_size; i < seq*step_size + overhang; i++) {
            if (maxVal < vec[i]) {
                max_idx = i + start;
                maxVal = vec[i];
            }
        }
        return {max_idx, maxVal};
    }
}

int max_elem_avx512_3_template(float * vec, size_t size) {
    auto [ idx, _] = max_elem_avx512_3_recur<__m512, __mmask16>(vec, size, 0, vec[0], 0);
    return idx;
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

#ifdef __SSE__
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
     std::cout << "This is a test that emulates finding the max_element from an array with a random size of up to " << MAX_OUTPUT_LAYER_SIZE
    << " elements.\nThe numbers in this array are floats drawn from an uniform distribution from -5:5. The test is run " << ITERATIONS
    << " times.\nThe purpose is to emulate a neural network output layer.\n"
    << "The algorithms presented will do very poorly if the array is sorted (or almost sorted) in ascending order, but will be very good in all other cases. Testing..." << std::endl;
    srand (time(NULL));
    std::chrono::duration<double> elapsed_seconds(0);
    std::chrono::duration<double> elapsed_seconds_seq(0);
#ifdef __AVX512F__
    std::chrono::duration<double> elapsed_seconds_avx512(0);
    std::chrono::duration<double> elapsed_seconds_avx512_2(0);
    std::chrono::duration<double> elapsed_seconds_avx512_3(0);
    std::chrono::duration<double> elapsed_seconds_avx512_4(0);
#endif
#ifdef __AVX__
    std::chrono::duration<double> elapsed_seconds_avx_3(0);
#endif
#ifdef __SSE__
    std::chrono::duration<double> elapsed_seconds_sse_3(0);
#endif
    for (int i = 0; i < ITERATIONS; i++) {
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

        // avx512, forth attempt
        auto start6 = std::chrono::steady_clock::now();
        int mymax6 = max_elem_avx512_3_template(logits, size);
        auto end6 = std::chrono::steady_clock::now();

        // Check for correctness
        if (!allEqual(mymax, mymax2, mymax3, mymax4, mymax5, mymax6)) {
            std::cerr << "Mymax1: " << logits[mymax] << " MyMax2 " << logits[mymax2]  << 
            " MyMax3 " << logits[mymax3] << " MyMax4 " << logits[mymax4] 
            << " MyMax5 " << logits[mymax5] <<" MyMax6 " << logits[mymax6] << " size: " << size << std::endl;
            break;
        }
        elapsed_seconds_avx512 += end3-start3;
        elapsed_seconds_avx512_2 += end4-start4;
        elapsed_seconds_avx512_3 += end5-start5;
        elapsed_seconds_avx512_4 += end6-start6;
#endif

#ifdef __AVX__
        auto start7 = std::chrono::steady_clock::now();
        int mymax7 = max_elem_avx_3(logits, size);
        auto end7 = std::chrono::steady_clock::now();

        // Check for correctness
        if (!allEqual(mymax, mymax2, mymax7)) {
            std::cerr << "Mymax1: " << logits[mymax] << "Mymax2: " << logits[mymax2] << " MyMaxAVX " << logits[mymax7]  <<  " size: " << size << std::endl;
            break;
        }
        elapsed_seconds_avx_3 += end7-start7;
#endif

#ifdef __SSE__
        auto start8 = std::chrono::steady_clock::now();
        int mymax8 = max_elem_sse_3(logits, size);
        auto end8 = std::chrono::steady_clock::now();

        // Check for correctness
        if (!allEqual(mymax, mymax8)) {
            std::cerr << "Mymax1: " << logits[mymax] << "Mymax2: " << logits[mymax2] <<  " MyMaxSSE " << logits[mymax8]  <<  " size: " << size << std::endl;
            break;
        }
        elapsed_seconds_sse_3 += end8-start8;
#endif
        free(logits);
    }
    std::cout << "Elapsed time. Baselines: \n"
    << "std::max_element:        "<< elapsed_seconds.count() << "s\n"
    << "simple_seq:              " << elapsed_seconds_seq.count() << "s\n"
#ifdef __AVX512F__
    << "AVX512F:\n"
    << "max + max_reduce:        "<< elapsed_seconds_avx512.count() << "s\n"
    << "max_reduce only:         "<< elapsed_seconds_avx512_2.count() << "s\n"
    << "cmp_ps_mask only:        "<< elapsed_seconds_avx512_3.count() << "s\n"
    << "^ + vectorised overhang: "<< elapsed_seconds_avx512_4.count() << "s\n"
#endif
#ifdef __AVX__
    << "AVX:\n"
    << "cmp_ps + move mask:      "<< elapsed_seconds_avx_3.count() << "s\n"
#endif
#ifdef __SSE__
    << "SSE:\n"
    << "cmplt_ps + move mask:    "<< elapsed_seconds_sse_3.count() << "s\n";
#else
;
#endif
    return 0;
}
