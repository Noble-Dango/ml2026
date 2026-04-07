/*
Author: Yousef Ammar

This file includes a custom implementation of pdqsort in which I cut out C++
features and restrict the type of keys to u32.
Credits to Orson Peters <orsonpeters@gmail.com>.
See https://github.com/orlp/pdqsort
and https://arxiv.org/abs/2106.05123.
The original source code for pdqsort is provided under the terms of the zlib
license.

The structure of the multithreaded section of the code is adapted from Ryan
Fleury's own. See https://www.dgtlgrove.com/p/multi-core-by-default.

I disclaim copyright to my own code.
In place of a legal notice, here is a short poem:

    I long to feel no shame
    As I look up at the sky.
    A mere breeze,
    Stirring the leaves,
    Pained my heart.
    With a heart that sings of the stars,
    I’ll love all dying things.
    And I must fare the path
    That’s been allotted to me.
    Tonight, also,
    The winds sweep over the stars.

    --Yoon Dong Ju, Prelude

*****
GENERAL COMMENTS

Note 0:
The file is split into sections marked by comments. Searching for "SECTION"
should help you navigate.

Note 1:
The flag FEATURE_MULTITHREADED can be passed to the compiler driver. It is used
to selectively compile parts of the code and to turn on or off multithreading.
While the code appears to have some redundant parts, these never get compiled.

Note 2:
The code uses neither locking nor atomic instructions. Because of the layout of
the data, no two threads write to the same memory.
I believe it's a neat solution.

Note 3:
'The IEE754 standard was deliberately made such that positive floating-point
numbers can be sorted as integers (both in the 32 and 64 bit format).'
(Source:
https://www.researchgate.net/publication/342347690_RadixInsert_a_much_faster_stable_algorithm_for_sorting_floating-point_numbers)

A given euclidean distance can only ever be positive.
Therefore, we may sort the array of distances as if they were integers.

Note 4:
A full sort of the keys (distances) is not employed, instead I use the partition
routine to implement quickselect (the pseudocode of which is taken from
wikipedia).

Nevertheless, I used sort_keys (which calls pdqsort_loop) in the beginning to
develop the classifier and later for testing.
*/

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef uint32_t u32;
typedef uint64_t u64;

#if !defined(THREAD_COUNT)
#define THREAD_COUNT 8
#endif

#define KiB(n) (n * 1024)
#define MiB(n) (KiB(n) * 1024)
#define GiB(n) (MiB(n) * 1024)

#define INIT_SIZE MiB(1)
#define ALIGN_PAGE(n) (n % KiB(4) ? n + (KiB(4) - n % KiB(4)) : n)

/**********************************************/
/* SECTION PUBLIC API */
/**********************************************/

// These can be used with arbitrary input (not necessarily owned by this
// process)
u32 *quickselect(u32 *keys, size_t len, size_t count_k, u64 *indices);
bool sort_keys(u32 *keys, size_t len, u64 *indices);

// You must call fit before predict:
int fit(double *source, int64_t *labels, size_t width, size_t height_src,
        size_t max_label);

// Returns a view into the labels determined by the classifier. This view lives
// and is valid only until the next call to predict. If a longer lifetime is
// desired, user must copy data.
int64_t *predict(double *new_data, size_t width, size_t height_new,
                 size_t k_limit);

/**********************************************/
/* SECTION DISTANCE CALCULATION */
/**********************************************/
#include <emmintrin.h>

#define SQUARE(r) ((r) * (r))

void distance_loop(float *distances, double *source, double *input,
                   size_t width, size_t height_src) {
    size_t height_by_four = height_src / 4;
    size_t height_rest_four = height_src - height_by_four * 4;

    for (size_t i = 0; i < height_by_four * 4; i += 4) {
        double *first_vec = source + i * width;
        double *second_vec = source + (i + 1) * width;
        double *third_vec = source + (i + 2) * width;
        double *fourth_vec = source + (i + 3) * width;

        double agg_one = 0.0;
        double agg_two = 0.0;
        double agg_three = 0.0;
        double agg_four = 0.0;

        // NOTE(yousef): I do not expect the features vector (the innermost) to
        // be so wide as to benefit from manual vectorisation.
        for (size_t l = 0; l < width; ++l) {
            double f_0 = first_vec[l] - input[l];
            f_0 = SQUARE(f_0);
            agg_one += f_0;

            double g_0 = second_vec[l] - input[l];
            g_0 = SQUARE(g_0);
            agg_two += g_0;

            double h_0 = third_vec[l] - input[l];
            h_0 = SQUARE(h_0);
            agg_three += h_0;

            double j_0 = fourth_vec[l] - input[l];
            j_0 = SQUARE(j_0);
            agg_four += j_0;
        }

        __m128 packed_singles =
            _mm_set_ps(agg_four, agg_three, agg_two, agg_one);

        __m128 packed_sqrts = _mm_sqrt_ps(packed_singles); // losing precision

        float out[4];
        _mm_storeu_ps(out, packed_sqrts);

        distances[i] = (float)out[0];
        distances[i + 1] = (float)out[1];
        distances[i + 2] = (float)out[2];
        distances[i + 3] = (float)out[3];
    }

    for (size_t i = height_by_four * 4; i < height_src; ++i) {
        double *current = source + i * width;
        double agg = 0.0;

        for (size_t k = 0; k < width; ++k) {
            double f_0 = current[k] - input[k];
            f_0 = SQUARE(f_0);
            agg += f_0;
        }

        __m128d scalar_agg = _mm_set_sd(agg);
        __m128d sqrt_agg = _mm_sqrt_sd(scalar_agg, scalar_agg);

        double result = _mm_cvtsd_f64(sqrt_agg);

        distances[i] = (float)result;
    }
}

/**********************************************/
/* SECTION KNN-CLASSIFIER */
/**********************************************/
struct {
    char *stack;
    size_t size;
} knn_memory = {0};

struct knn_classifier_t {
    double *data;
    int64_t *labels;

    size_t max_label;
    size_t height_src;
    size_t width_src;

    char *thread_mem[THREAD_COUNT];

    double *new_data;

    int64_t *results;
} classifier = {0};

static void init_knn() {
    size_t base = MiB(50);
    size_t size = THREAD_COUNT * MiB(20) + base;
    char *res = (char *)malloc(size);

    // touch memory
    for (size_t i = 0; i < size; ++i) {
        res[i] = 0;
    }

    knn_memory.stack = res;
    knn_memory.size = size;

    classifier.data = (double *)knn_memory.stack;
    classifier.labels = (int64_t *)(knn_memory.stack + MiB(10));

    classifier.new_data = (double *)(knn_memory.stack + MiB(20));

    classifier.results = (int64_t *)(knn_memory.stack + MiB(30));

    for (size_t i = 0; i < THREAD_COUNT; ++i) {
        classifier.thread_mem[i] = knn_memory.stack + base + i * MiB(20);
    }
}

int fit(double *source, int64_t *labels, size_t width, size_t height_src,
        size_t max_label) {
    init_knn();

    classifier.height_src = height_src;
    classifier.width_src = width;

    classifier.max_label = max_label;

    memcpy(classifier.data, source, height_src * width * sizeof(double));
    memcpy(classifier.labels, labels, height_src * sizeof(int64_t));

    return (1);
}

struct predict_params_t {
    double *new_data;
    size_t width;
    size_t height_new;
    size_t k_limit;
    size_t thread_idx;
};

void predict_multithreaded(struct predict_params_t params);

#ifdef _WIN32
#include <Windows.h>
DWORD WINAPI StartWorkWin32(LPVOID lpParameter) {
    predict_multithreaded(*(struct predict_params_t *)lpParameter);
}
#else
#include <pthread.h>
void *StartWorkPOSIX(void *params) {
    predict_multithreaded(*(struct predict_params_t *)params);
    return NULL;
}
#endif

// correct labels stored at classifier.results
void predict_multithreaded(struct predict_params_t params) {
    size_t thread_idx = params.thread_idx;

    char *memory = classifier.thread_mem[thread_idx];
    float *distance_mem = memory;
    size_t *label_counts = memory + MiB(5);
    int64_t *k_labels = memory + MiB(5) + KiB(200);
    u64 *indices = memory + MiB(5) + KiB(400);

    size_t routines_count = params.height_new;
    size_t routines_per_thread = routines_count / THREAD_COUNT;
    size_t leftover_routines = routines_count % THREAD_COUNT;

    size_t first_vector_this_thread = routines_per_thread * thread_idx;
    size_t one_past_last_vector_this_thread =
        first_vector_this_thread + routines_per_thread;

    for (size_t i = first_vector_this_thread;
         i < one_past_last_vector_this_thread; ++i) {
        memset(label_counts, '\0', classifier.max_label * sizeof(size_t));

        double *vector = params.new_data + i * params.width;

        distance_loop(distance_mem, classifier.data, vector,
                      classifier.width_src, classifier.height_src);

        // must compile with fno-strict-aliasing for now
        u32 *keys = (u32 *)distance_mem;
        quickselect(keys, classifier.height_src, params.k_limit, indices);

        for (size_t u = 0; u < params.k_limit; ++u) {
            u64 label_at = indices[u];
            k_labels[u] = classifier.labels[label_at];
        }

        size_t best_count = 0;
        int64_t best_label = (-1);

        for (size_t u = 0; u < params.k_limit; ++u) {
            int64_t label = k_labels[u];
            label_counts[label]++;

            if (label_counts[label] > best_count) {
                best_count = label_counts[label];
                best_label = label;
            }
        }

        classifier.results[i] = best_label;
    }

    if (thread_idx == 0) {
        for (size_t i = routines_per_thread * THREAD_COUNT; i < routines_count;
             ++i) {

            memset(label_counts, '\0', classifier.max_label * sizeof(size_t));

            double *vector = params.new_data + i * params.width;

            distance_loop(distance_mem, classifier.data, vector,
                          classifier.width_src, classifier.height_src);

            // must compile with fno-strict-aliasing for now
            u32 *keys = (u32 *)distance_mem;
            quickselect(keys, classifier.height_src, params.k_limit, indices);

            for (size_t u = 0; u < params.k_limit; ++u) {
                u64 label_at = indices[u];
                k_labels[u] = classifier.labels[label_at];
            }

            size_t best_count = 0;
            int64_t best_label = (-1);

            for (size_t u = 0; u < params.k_limit; ++u) {
                int64_t label = k_labels[u];
                label_counts[label]++;

                if (label_counts[label] > best_count) {
                    best_count = label_counts[label];
                    best_label = label;
                }
            }

            classifier.results[i] = best_label;
        }
    }
}

int64_t *predict(double *new_data, size_t width, size_t height_new,
                 size_t k_limit) {
    memcpy(classifier.new_data, new_data, height_new * width * sizeof(double));

#if defined(FEATURE_MULTITHREADED)
#if defined(_WIN32)
    HANDLE threads[THREAD_COUNT] = {0};

    struct predict_params_t parameters[THREAD_COUNT] = {0};

    for (size_t thread_idx = 0; thread_idx < THREAD_COUNT; thread_idx += 1) {
        parameters[thread_idx].new_data = classifier.new_data;
        parameters[thread_idx].width = width;
        parameters[thread_idx].height_new = height_new;
        parameters[thread_idx].k_limit = k_limit;
        parameters[thread_idx].thread_idx = thread_idx;

        DWORD ThreadID;
        threads[thread_idx] = CreateThread(
            0, 0, StartWorkWin32, &parameters[thread_idx], 0, &ThreadID);
    }

    for (size_t thread_idx = 0; thread_idx < THREAD_COUNT; thread_idx += 1) {
        WaitForMultipleObjects(THREAD_COUNT, threads, true, INFINITE);
    }
#else  // POSIX: https://pages.cs.wisc.edu/~remzi/OSTEP/threads-api.pdf
    pthread_t threads[THREAD_COUNT] = {0};

    struct predict_params_t parameters[THREAD_COUNT] = {0};

    for (size_t thread_idx = 0; thread_idx < THREAD_COUNT; thread_idx += 1) {
        parameters[thread_idx].new_data = classifier.new_data;
        parameters[thread_idx].width = width;
        parameters[thread_idx].height_new = height_new;
        parameters[thread_idx].k_limit = k_limit;
        parameters[thread_idx].thread_idx = thread_idx;

        threads[thread_idx] = (pthread_t){0};
        int res = pthread_create(&threads[thread_idx], NULL, StartWorkPOSIX,
                                 &parameters[thread_idx]);
    }

    for (size_t thread_idx = 0; thread_idx < THREAD_COUNT; thread_idx += 1) {
        pthread_join(threads[thread_idx], NULL);
    }
#endif // win or posix
#else  // single threaded
    char *memory = classifier.thread_mem[0];
    float *distance_mem = memory;
    size_t *label_counts = memory + MiB(5);
    int64_t *k_labels = memory + MiB(5) + KiB(200);
    u64 *indices = memory + MiB(5) + KiB(400);

    for (size_t i = 0; i < height_new; ++i) {
        memset(label_counts, '\0', classifier.max_label * sizeof(size_t));

        double *vector = new_data + i * width;

        distance_loop(distance_mem, classifier.data, vector,
                      classifier.width_src, classifier.height_src);

        // must compile with fno-strict-aliasing for now
        u32 *keys = (u32 *)distance_mem;
        quickselect(keys, classifier.height_src, k_limit, indices);

        for (size_t u = 0; u < k_limit; ++u) {
            u64 label_at = indices[u];
            k_labels[u] = classifier.labels[label_at];
        }

        size_t best_count = 0;
        int64_t best_label = (-1);

        for (size_t u = 0; u < k_limit; ++u) {
            int64_t label = k_labels[u];
            label_counts[label]++;

            if (label_counts[label] > best_count) {
                best_count = label_counts[label];
                best_label = label;
            }
        }

        classifier.results[i] = best_label;
    }
#endif
    return classifier.results;
}

/**********************************************/
/* SECTION SORTING */
/**********************************************/
#define INSERTION_SORT_THRESHOLD 24
#define NINTHER_THRESHOLD 128
#define PARTIAL_INSERTION_SORT_LIMIT 8

#define COMP(A, B) (A < B)

static void swap_u32(u32 *a, u32 *b, u32 *keys, u64 *indices) {
    u32 temp = *a;
    *a = *b;
    *b = temp;

    size_t index_left = (a - keys);
    size_t index_right = (b - keys);
    u64 tmp = *(indices + index_left);
    indices[index_left] = *(indices + index_right);
    indices[index_right] = tmp;
}

static void sort2(u32 *a, u32 *b, u32 *keys, u64 *indices) {
    if (COMP(*b, *a)) {
        swap_u32(a, b, keys, indices);
    }
}

static void sort3(u32 *a, u32 *b, u32 *c, u32 *keys, u64 *indices) {
    sort2(a, b, keys, indices);
    sort2(b, c, keys, indices);
    sort2(a, b, keys, indices);
}

static void insertion_sort(u32 *begin, u32 *end, u32 *keys, u64 *indices) {
    if (begin == end) {
        return;
    }

    for (u32 *cur = begin + 1; cur != end; ++cur) {
        u32 *sift = cur;
        u32 *sift_1 = cur - 1;

        if (COMP(*sift, *sift_1)) {
            u32 tmp = *sift;
            size_t sift_at = (sift - keys);
            u64 index = indices[sift_at];

            do {
                *sift = *sift_1;

                size_t sift_one_at = (sift_1 - keys);
                sift_at = (sift - keys);

                indices[sift_at] = *(indices + sift_one_at);

                sift--;
            } while (sift != begin && COMP(tmp, *--sift_1));

            *sift = tmp;
            sift_at = (sift - keys);
            indices[sift_at] = index;
        }
    }
};

static void unguarded_insertion_sort(u32 *begin, u32 *end, u32 *keys,
                                     u64 *indices) {
    if (begin == end) {
        return;
    }

    for (u32 *cur = begin + 1; cur != end; ++cur) {
        u32 *sift = cur;
        u32 *sift_1 = cur - 1;

        if (COMP(*sift, *sift_1)) {
            u32 tmp = *sift;
            size_t sift_at = (sift - keys);
            u64 index = indices[sift_at];

            do {
                *sift = *sift_1;

                size_t sift_one_at = (sift_1 - keys);
                sift_at = (sift - keys);

                indices[sift_at] = *(indices + sift_one_at);

                sift--;
            } while (COMP(tmp, *--sift_1));

            *sift = tmp;
            sift_at = (sift - keys);
            indices[sift_at] = index;
        }
    }
};

static bool partial_insertion_sort(u32 *begin, u32 *end, u32 *keys,
                                   u64 *indices) {
    if (begin == end) {
        return true;
    }

    size_t limit = 0;

    for (u32 *cur = begin + 1; cur != end; ++cur) {
        u32 *sift = cur;
        u32 *sift_1 = cur - 1;

        if (COMP(*sift, *sift_1)) {
            u32 tmp = *sift;
            size_t sift_at = (sift - keys);
            u64 index = indices[sift_at];

            do {
                *sift = *sift_1;

                size_t sift_one_at = (sift_1 - keys);
                sift_at = (sift - keys);

                indices[sift_at] = *(indices + sift_one_at);

                sift--;
            } while (sift != begin && COMP(tmp, *--sift_1));

            *sift = tmp;
            sift_at = (sift - keys);
            indices[sift_at] = index;

            limit += cur - sift;
        }

        if (limit > PARTIAL_INSERTION_SORT_LIMIT) {
            return false;
        }
    }

    return true;
};

struct partition_result {
    u32 *pivot_at;
    bool already_partitioned;
};

static struct partition_result partition_right(u32 *begin, u32 *end, u32 *keys,
                                               u64 *indices) {
    u32 pivot = *begin;
    size_t og_pivot_at = begin - keys;

    u32 *first = begin;
    u32 *last = end;

    // Find the first element greater than or equal than the pivot (the
    // median of 3 guarantees this exists).
    while (COMP(*++first, pivot)) {
    }

    // Find the first element strictly smaller than the pivot. We have to
    // guard this search if there was no element before *first.
    if (first - 1 == begin) {
        while (first < last && !COMP(*--last, pivot)) {
        }
    } else {
        while (!COMP(*--last, pivot)) {
        }
    }

    // If the first pair of elements that should be swapped to partition are
    // the same element, the passed in sequence already was correctly
    // partitioned.
    bool already_partitioned = first >= last;

    while (first < last) {
        swap_u32(first, last, keys, indices);
        while (COMP(*++first, pivot)) {
        }
        while (!COMP(*--last, pivot)) {
        }
    }

    u32 *pivot_pos = first - 1;

    *begin = *pivot_pos;
    *pivot_pos = pivot;

    size_t begin_at = begin - keys;
    size_t pivot_pos_at = pivot_pos - keys;

    u64 pivot_index = indices[og_pivot_at];
    indices[begin_at] = indices[pivot_pos_at];
    indices[pivot_pos_at] = pivot_index;

    return (struct partition_result){pivot_pos, already_partitioned};
}

// Similar function to the one above, except elements equal to the pivot are
// put to the left of the pivot and it doesn't check or return if the passed
// sequence already was partitioned. Since this is rarely used (the many
// equal case), and in that case pdqsort already has O(n) performance, no
// block quicksort is applied here for simplicity.
static u32 *partition_left(u32 *begin, u32 *end, u32 *keys, u64 *indices) {
    u32 pivot = *begin;
    size_t og_pivot_at = begin - keys;

    u32 *first = begin;
    u32 *last = end;

    while (COMP(pivot, *--last)) {
    }

    if (last + 1 == end) {
        while (first < last && !COMP(pivot, *++first)) {
        }
    } else {
        while (!COMP(pivot, *++first)) {
        }
    }

    while (first < last) {
        swap_u32(first, last, keys, indices);
        while (COMP(pivot, *--last)) {
        }
        while (!COMP(pivot, *++first)) {
        }
    }

    u32 *pivot_pos = last;
    *begin = *pivot_pos;
    *pivot_pos = pivot;

    size_t begin_at = begin - keys;
    size_t pivot_pos_at = pivot_pos - keys;

    u64 pivot_index = indices[og_pivot_at];
    indices[begin_at] = indices[pivot_pos_at];
    indices[pivot_pos_at] = pivot_index;

    return pivot_pos;
}

static void pdqsort_loop(u32 *begin, u32 *end, bool leftmost, u32 *keys,
                         u64 *indices) {
    while (true) {
        size_t size = end - begin;

        if (size < INSERTION_SORT_THRESHOLD) {
            if (leftmost) {
                insertion_sort(begin, end, keys, indices);
            } else {
                unguarded_insertion_sort(begin, end, keys, indices);
            }
            return;
        }

        size_t s2 = size / 2;
        if (size > NINTHER_THRESHOLD) {
            sort3(begin, begin + s2, end - 1, keys, indices);
            sort3(begin + 1, begin + (s2 - 1), end - 2, keys, indices);
            sort3(begin + 2, begin + (s2 + 1), end - 3, keys, indices);
            sort3(begin + (s2 - 1), begin + s2, begin + (s2 + 1), keys,
                  indices);

            swap_u32(begin, begin + s2, keys, indices);
        } else {
            sort3(begin + s2, begin, end - 1, keys, indices);
        }

        // If *(begin - 1) is the end of the right partition of a previous
        // partition operation there is no element in [begin, end) that is
        // smaller than *(begin - 1). Then if our pivot compares equal to
        // *(begin - 1) we change strategy, putting equal elements in the
        // left partition, greater elements in the right partition. We do
        // not have to recurse on the left partition, since it's sorted (all
        // equal).
        if (!leftmost && !COMP(*(begin - 1), *begin)) {
            begin = partition_left(begin, end, keys, indices) + 1;
            continue;
        }

        struct partition_result res =
            partition_right(begin, end, keys, indices);
        u32 *pivot_pos = res.pivot_at;
        bool already_partitioned = res.already_partitioned;

        size_t l_size = pivot_pos - begin;
        size_t r_size = end - (pivot_pos + 1);
        bool highly_unbalanced = l_size < size / 8 || r_size < size / 8;

        if (highly_unbalanced) {
            // TODO(yousef): bad_allowed == 0? then sort_heap

            if (l_size >= INSERTION_SORT_THRESHOLD) {
                swap_u32(begin, begin + l_size / 4, keys, indices);
                swap_u32(pivot_pos - 1, pivot_pos - l_size / 4, keys, indices);

                if (l_size > NINTHER_THRESHOLD) {
                    swap_u32(begin + 1, begin + (l_size / 4 + 1), keys,
                             indices);
                    swap_u32(begin + 2, begin + (l_size / 4 + 2), keys,
                             indices);
                    swap_u32(pivot_pos - 2, pivot_pos - (l_size / 4 + 1), keys,
                             indices);
                    swap_u32(pivot_pos - 3, pivot_pos - (l_size / 4 + 2), keys,
                             indices);
                }
            }

            if (r_size >= INSERTION_SORT_THRESHOLD) {
                swap_u32(pivot_pos + 1, pivot_pos + (1 + r_size / 4), keys,
                         indices);
                swap_u32(end - 1, end - r_size / 4, keys, indices);

                if (r_size > NINTHER_THRESHOLD) {
                    swap_u32(pivot_pos + 2, pivot_pos + (2 + r_size / 4), keys,
                             indices);
                    swap_u32(pivot_pos + 3, pivot_pos + (3 + r_size / 4), keys,
                             indices);
                    swap_u32(end - 2, end - (1 + r_size / 4), keys, indices);
                    swap_u32(end - 3, end - (2 + r_size / 4), keys, indices);
                }
            }

        } else {
            // If we were decently balanced and we tried to sort an already
            // partitioned sequence try to use insertion sort.
            if (already_partitioned &&
                partial_insertion_sort(begin, pivot_pos, keys, indices) &&
                partial_insertion_sort(pivot_pos + 1, end, keys, indices))
                return;
        }

        // Sort the left partition first using recursion and do tail
        // recursion elimination for the right-hand partition.
        pdqsort_loop(begin, pivot_pos, leftmost, keys, indices);
        begin = pivot_pos + 1;
        leftmost = false;
    }
}

static void init_sorter(size_t len, u32 *keys, u64 *indices) {
    for (u64 i = 0; i < len; ++i) {
        indices[i] = i;
    }
}

bool sort_keys(u32 *keys, size_t len, u64 *indices) {
    init_sorter(len, keys, indices);

    u32 *begin = keys;
    u32 *end = keys + len;

    if (begin == end) {
        return (true);
    }

#define INIT_PDQSORT_CALL true
    pdqsort_loop(begin, end, INIT_PDQSORT_CALL, keys, indices);

    return true;
}

// quickselect: returns pointer to the kth element
// for all e between base and &k: e =< k
u32 *quickselect(u32 *keys, size_t len, size_t count_k, u64 *indices) {

    u32 *left = keys;
    u32 *right = keys + len;

    init_sorter(len, keys, indices);

    // TODO(yousef): how about shuffling things like in pdqsort_loop before
    // choosing a pivot?

    while (true) {
        if (left == right) {
            return left;
        }

        u32 *pivot = partition_left(left, right, keys, indices);

        size_t pivot_index = pivot - keys;

        if (count_k == pivot_index) {
            return pivot;
        } else if (count_k < pivot_index) {
            right -= 1;
        } else {
            left += 1;
        }
    }
}
