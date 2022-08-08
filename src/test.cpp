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
    float maxVal = vec[0]; // Ignore empty vec
    int max_idx = 0;
    for (size_t i = 0; i < size; i++) {
        if (maxVal > vec[i]) {
            max_idx = i;
        }
    }
    return max_idx;
}

int max_elem_avx512(float * vec, size_t size) {
    float maxVal = vec[0];
    int max_idx = 0;
    div_t setup = div(size, 32);
    int overhang = setup.rem;
    int seq = setup.quot;
    for (size_t i = 0; i < seq; i+=32) {
        __m512 first = _mm512_load_ps(&vec[i]);
        __m512 second = _mm512_load_ps(&vec[i+16]);
        __m512 maxd = _mm512_max_ps(first, second);
        float max_single = _mm512_reduce_max_ps(maxd);
        if (max_single > maxVal) {
            /* @ todo INTRICATE IT
            __mm512 maxvalVec = _mm512_set1_ps(maxVal);
            __mmask16 comparison = _mm512_cmp_ps_mask(first, maxvalVec, 0);
            int newidx = 0; //@TODO extract the non zero bit of the mask
            if (comparison != 0) {
                // EXTRACT the non zero bit of the mask
            } else {
                comparison = _mm512_cmp_ps_mask(first, maxvalVec, 0);
                // Extract the non-zero bit of the mask.
            }*/
            for (int j = 0; j<32; j++) {
                if (vec[j] == max_single) {
                    maxVal = vec[j];
                    max_idx = i+j;
                }
            }
        }
    }

    for (int i = seq; i<size; i++) {
        if (maxVal > vec[i]) {
            max_idx = i;
        }
    }
    return max_idx;
}

int main() {
    std::chrono::duration<double> elapsed_seconds(0);
    std::chrono::duration<double> elapsed_seconds_seq(0);
    std::chrono::duration<double> elapsed_seconds_avx512(0);
    for (int i = 0; i < 1000000; i++) {
            size_t size = getSize();
        float *logits = (float *)aligned_alloc(1024, size*sizeof(float));
        populateVec(logits, size);
        auto start = std::chrono::steady_clock::now();
        int mymax = max_elem(logits, size);
        auto end = std::chrono::steady_clock::now();

        // Mine:
        auto start2 = std::chrono::steady_clock::now();
        int mymax2 = max_elem_manual(logits, size);
        auto end2 = std::chrono::steady_clock::now();

        // avx512, first attempt
        auto start3 = std::chrono::steady_clock::now();
        int mymax3 = max_elem_avx512(logits, size);
        auto end3 = std::chrono::steady_clock::now();

        if (!allEqual(mymax, mymax2, mymax3)) {
            break;
        }

        //@TODO check for correctness
        //std::cout << "Max: " << mymax << " Size: " << size << std::endl;
        elapsed_seconds += end-start;
        elapsed_seconds_seq += end2-start2;
        elapsed_seconds_avx512 += end3-start3;
        free(logits);
    }
    std::cout << "Elapsed time:\n"
    << "std::max_element: "<< elapsed_seconds.count() << "s\n"
    << "max_element_avx512: "<< elapsed_seconds_avx512.count() << "s\n"
    << "simple_seq: " << elapsed_seconds_seq.count() << "s" << std::endl;
    return 0;
}
