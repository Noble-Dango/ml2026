/*
This file includes a custom implementation of pdqsort in which I cut out C++
features and restrict the type of keys to u32.

Credits to Orson Peters <orsonpeters@gmail.com>.
See https://github.com/orlp/pdqsort
and https://arxiv.org/abs/2106.05123.
The original source code is provided under the terms of the zlib license.

Note 1:
The IEE754 standard was deliberately made such that positive floating-point
numbers can be sorted as integers (both in the 32 and 64 bit format).
https://www.researchgate.net/publication/342347690_RadixInsert_a_much_faster_stable_algorithm_for_sorting_floating-point_numbers

A given euclidean distance can only ever be positive.
Therefore, we may sort the array of distances as if they were integers.

Note 2:
In the original pdqsort source code, Compare is used.

https://en.cppreference.com/w/cpp/named_req/Compare.html
https://en.wikipedia.org/wiki/Weak_ordering#Strict_weak_orderings

The return value of the function call operation applied to an object of a type
satisfying Compare, when converted to bool, yields (1) true if the first
argument of the call appears before the second in the strict weak ordering
relation induced by this type, and false otherwise.

Establishes strict weak ordering relation with the following properties:
For all a, comp(a, a) == false.
If comp(a, b) == true then comp(b, a) == false.
if comp(a, b) == true and comp(b, c) == true then comp(a, c) == true.

In our case, this is sufficient for u32:
comp(a,b) := a < b
*/

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define INSERTION_SORT_THRESHOLD 24
#define NINTHER_THRESHOLD 128
#define PARTIAL_INSERTION_SORT_LIMIT 8

typedef uint32_t u32;
typedef uint64_t u64;

#define COMP(A, B) (A < B)

struct sorter_t {
    u64 *indices;
    // strictly speaking, capacity != occupied, but this is
    // enough to prevent reading outside bounds (in view_indices and
    // copy_indices), and that's what I care about
    size_t capacity;
    u32 *keys;
};

struct sorter_t sorter = {0};

void swap_u32(u32 *a, u32 *b) {
    u32 temp = *a;
    *a = *b;
    *b = temp;

    size_t index_left = (a - sorter.keys);
    size_t index_right = (b - sorter.keys);
    u64 tmp = *(sorter.indices + index_left);
    sorter.indices[index_left] = *(sorter.indices + index_right);
    sorter.indices[index_right] = tmp;
}

void sort2(u32 *a, u32 *b) {
    if (COMP(*b, *a)) {
        swap_u32(a, b);
    }
}

void sort3(u32 *a, u32 *b, u32 *c) {
    sort2(a, b);
    sort2(b, c);
    sort2(a, b);
}

void insertion_sort(u32 *begin, u32 *end) {
    if (begin == end) {
        return;
    }

    for (u32 *cur = begin + 1; cur != end; ++cur) {
        u32 *sift = cur;
        u32 *sift_1 = cur - 1;

        if (COMP(*sift, *sift_1)) {
            u32 tmp = *sift;
            size_t sift_at = (sift - sorter.keys);
            u64 index = sorter.indices[sift_at];

            do {
                *sift = *sift_1;

                size_t sift_one_at = (sift_1 - sorter.keys);
                sift_at = (sift - sorter.keys);

                sorter.indices[sift_at] = *(sorter.indices + sift_one_at);

                sift--;
            } while (sift != begin && COMP(tmp, *--sift_1));

            *sift = tmp;
            sift_at = (sift - sorter.keys);
            sorter.indices[sift_at] = index;
        }
    }
};

void unguarded_insertion_sort(u32 *begin, u32 *end) {
    if (begin == end) {
        return;
    }

    for (u32 *cur = begin + 1; cur != end; ++cur) {
        u32 *sift = cur;
        u32 *sift_1 = cur - 1;

        if (COMP(*sift, *sift_1)) {
            u32 tmp = *sift;
            size_t sift_at = (sift - sorter.keys);
            u64 index = sorter.indices[sift_at];

            do {
                *sift = *sift_1;

                size_t sift_one_at = (sift_1 - sorter.keys);
                sift_at = (sift - sorter.keys);

                sorter.indices[sift_at] = *(sorter.indices + sift_one_at);

                sift--;
            } while (COMP(tmp, *--sift_1));

            *sift = tmp;
            sift_at = (sift - sorter.keys);
            sorter.indices[sift_at] = index;
        }
    }
};

bool partial_insertion_sort(u32 *begin, u32 *end) {
    if (begin == end) {
        return true;
    }

    size_t limit = 0;

    for (u32 *cur = begin + 1; cur != end; ++cur) {
        u32 *sift = cur;
        u32 *sift_1 = cur - 1;

        if (COMP(*sift, *sift_1)) {
            u32 tmp = *sift;
            size_t sift_at = (sift - sorter.keys);
            u64 index = sorter.indices[sift_at];

            do {
                *sift = *sift_1;

                size_t sift_one_at = (sift_1 - sorter.keys);
                sift_at = (sift - sorter.keys);

                sorter.indices[sift_at] = *(sorter.indices + sift_one_at);

                sift--;
            } while (sift != begin && COMP(tmp, *--sift_1));

            *sift = tmp;
            sift_at = (sift - sorter.keys);
            sorter.indices[sift_at] = index;

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

struct partition_result partition_right(u32 *begin, u32 *end) {
    u32 pivot = *begin;
    size_t og_pivot_at = begin - sorter.keys;

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
        swap_u32(first, last);
        while (COMP(*++first, pivot)) {
        }
        while (!COMP(*--last, pivot)) {
        }
    }

    u32 *pivot_pos = first - 1;

    *begin = *pivot_pos;
    *pivot_pos = pivot;

    size_t begin_at = begin - sorter.keys;
    size_t pivot_pos_at = pivot_pos - sorter.keys;

    u64 pivot_index = sorter.indices[og_pivot_at];
    sorter.indices[begin_at] = sorter.indices[pivot_pos_at];
    sorter.indices[pivot_pos_at] = pivot_index;

    return (struct partition_result){pivot_pos, already_partitioned};
}

// Similar function to the one above, except elements equal to the pivot are
// put to the left of the pivot and it doesn't check or return if the passed
// sequence already was partitioned. Since this is rarely used (the many
// equal case), and in that case pdqsort already has O(n) performance, no
// block quicksort is applied here for simplicity.
u32 *partition_left(u32 *begin, u32 *end) {
    u32 pivot = *begin;
    size_t og_pivot_at = begin - sorter.keys;

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
        swap_u32(first, last);
        while (COMP(pivot, *--last)) {
        }
        while (!COMP(pivot, *++first)) {
        }
    }

    u32 *pivot_pos = last;
    *begin = *pivot_pos;
    *pivot_pos = pivot;

    size_t begin_at = begin - sorter.keys;
    size_t pivot_pos_at = pivot_pos - sorter.keys;

    u64 pivot_index = sorter.indices[og_pivot_at];
    sorter.indices[begin_at] = sorter.indices[pivot_pos_at];
    sorter.indices[pivot_pos_at] = pivot_index;

    return pivot_pos;
}

void pdqsort_loop(u32 *begin, u32 *end, bool leftmost) {
    while (true) {
        size_t size = end - begin;

        if (size < INSERTION_SORT_THRESHOLD) {
            if (leftmost) {
                insertion_sort(begin, end);
            } else {
                unguarded_insertion_sort(begin, end);
            }
            return;
        }

        size_t s2 = size / 2;
        if (size > NINTHER_THRESHOLD) {
            sort3(begin, begin + s2, end - 1);
            sort3(begin + 1, begin + (s2 - 1), end - 2);
            sort3(begin + 2, begin + (s2 + 1), end - 3);
            sort3(begin + (s2 - 1), begin + s2, begin + (s2 + 1));

            swap_u32(begin, begin + s2);
        } else {
            sort3(begin + s2, begin, end - 1);
        }

        // If *(begin - 1) is the end of the right partition of a previous
        // partition operation there is no element in [begin, end) that is
        // smaller than *(begin - 1). Then if our pivot compares equal to
        // *(begin - 1) we change strategy, putting equal elements in the
        // left partition, greater elements in the right partition. We do
        // not have to recurse on the left partition, since it's sorted (all
        // equal).
        if (!leftmost && !COMP(*(begin - 1), *begin)) {
            begin = partition_left(begin, end) + 1;
            continue;
        }

        struct partition_result res = partition_right(begin, end);
        u32 *pivot_pos = res.pivot_at;
        bool already_partitioned = res.already_partitioned;

        size_t l_size = pivot_pos - begin;
        size_t r_size = end - (pivot_pos + 1);
        bool highly_unbalanced = l_size < size / 8 || r_size < size / 8;

        if (highly_unbalanced) {
            // TODO(yousef): bad_allowed == 0? then sort_heap

            if (l_size >= INSERTION_SORT_THRESHOLD) {
                swap_u32(begin, begin + l_size / 4);
                swap_u32(pivot_pos - 1, pivot_pos - l_size / 4);

                if (l_size > NINTHER_THRESHOLD) {
                    swap_u32(begin + 1, begin + (l_size / 4 + 1));
                    swap_u32(begin + 2, begin + (l_size / 4 + 2));
                    swap_u32(pivot_pos - 2, pivot_pos - (l_size / 4 + 1));
                    swap_u32(pivot_pos - 3, pivot_pos - (l_size / 4 + 2));
                }
            }

            if (r_size >= INSERTION_SORT_THRESHOLD) {
                swap_u32(pivot_pos + 1, pivot_pos + (1 + r_size / 4));
                swap_u32(end - 1, end - r_size / 4);

                if (r_size > NINTHER_THRESHOLD) {
                    swap_u32(pivot_pos + 2, pivot_pos + (2 + r_size / 4));
                    swap_u32(pivot_pos + 3, pivot_pos + (3 + r_size / 4));
                    swap_u32(end - 2, end - (1 + r_size / 4));
                    swap_u32(end - 3, end - (2 + r_size / 4));
                }
            }

        } else {
            // If we were decently balanced and we tried to sort an already
            // partitioned sequence try to use insertion sort.
            if (already_partitioned &&
                partial_insertion_sort(begin, pivot_pos) &&
                partial_insertion_sort(pivot_pos + 1, end))
                return;
        }

        // Sort the left partition first using recursion and do tail
        // recursion elimination for the right-hand partition.
        pdqsort_loop(begin, pivot_pos, leftmost);
        begin = pivot_pos + 1;
        leftmost = false;
    }
}

void pdqsort(u32 *begin, u32 *end) {
    if (begin == end)
        return;

#define INIT_PDQSORT_CALL true
    pdqsort_loop(begin, end, INIT_PDQSORT_CALL);
}

#define KiB(n) (n * 1024)
#define MiB(n) (KiB(n) * 1024)
#define GiB(n) (MiB(n) * 1024)

#define INIT_SIZE MiB(1)
#define ALIGN_PAGE(n) (n % KiB(4) ? n + (KiB(4) - n % KiB(4)) : n)

bool sort_keys(u32 *keys, size_t len) {
    sorter.keys = keys;

    if (sorter.indices == NULL) {
        size_t require = sizeof(u64) * len;

        size_t size = INIT_SIZE > require ? INIT_SIZE : require;

        size = ALIGN_PAGE(size);

        void *res = malloc(size);
        if (res == NULL) {
            return false;
        }

        sorter.indices = (u64 *)res;
        sorter.capacity = size / sizeof(u64);
    } else if (sorter.capacity < len) {
        size_t require = ALIGN_PAGE(sizeof(u64) * len);

        void *res = realloc(sorter.indices, require);
        if (res == NULL) {
            return false;
        }

        sorter.indices = res;
        sorter.capacity = require / sizeof(u64);
    }

    for (size_t i = 0; i < sorter.capacity; ++i) {
        sorter.indices[i] = i;
    }

    pdqsort(sorter.keys, sorter.keys + len);
    return true;
}

bool view_internal(u64 **indices, size_t how_many) {
    if (sorter.indices == NULL) {
        return false;
    }
    if (how_many > sorter.capacity) {
        return false;
    }

    *indices = sorter.indices;

    return true;
}

bool copy_indices(u64 *dest_container, size_t how_many) {
    if (sorter.indices == NULL) {
        return false;
    }

    if (how_many > sorter.capacity) {
        return false;
    }

    memcpy(dest_container, sorter.indices, how_many * sizeof(u64));

    return true;
}

void deinitialise_sorter() {
    free(sorter.indices);
    sorter = (struct sorter_t){0};
};

#define LEN(ARR) (sizeof(ARR) / sizeof((ARR)[0]))

#include <emmintrin.h>

#define SQUARE(r) ((r) * (r))

void distance_loop(float *distances, double *source, double *input,
                   size_t width, size_t height_src) {

    size_t by_two = height_src / 2;
    size_t rest_two = height_src - by_two * 2;

    for (size_t i = 0; i < by_two * 2; i += 2) {
        double agg_one = 0.0;
        double agg_two = 0.0;

        size_t by_four = width / 4;

        double *first_vec = source + i * width;
        double *second_vec = source + (i + 1) * width;

        for (size_t k = 0; k < by_four * 4; k += 4) {
            double f_0 = first_vec[k + 0] - input[k + 0];
            f_0 = SQUARE(f_0);
            agg_one += f_0;

            double f_1 = first_vec[k + 1] - input[k + 1];
            f_1 = SQUARE(f_1);
            agg_one += f_1;

            double f_2 = first_vec[k + 2] - input[k + 2];
            f_2 = SQUARE(f_2);
            agg_one += f_2;

            double f_3 = first_vec[k + 3] - input[k + 3];
            f_3 = SQUARE(f_3);
            agg_one += f_3;

            double r_0 = second_vec[k + 0] - input[k + 0];
            r_0 = SQUARE(r_0);
            agg_two += r_0;

            double r_1 = second_vec[k + 1] - input[k + 1];
            r_1 = SQUARE(r_1);
            agg_two += r_1;

            double r_2 = second_vec[k + 2] - input[k + 2];
            r_2 = SQUARE(r_2);
            agg_two += r_2;

            double r_3 = second_vec[k + 3] - input[k + 3];
            r_3 = SQUARE(r_3);
            agg_two += r_3;
        }

        for (size_t l = by_four * 4; l < width; ++l) {
            double f_0 = first_vec[l] - input[l];
            f_0 = SQUARE(f_0);
            agg_one += f_0;

            double r_0 = second_vec[l] - input[l];
            r_0 = SQUARE(r_0);
            agg_two += r_0;
        }

        __m128d packed_two = _mm_set_pd(agg_two, agg_one);
        __m128d packed_sqrts = _mm_sqrt_pd(packed_two);

        double out[2];
        _mm_storeu_pd(out, packed_sqrts);

        distances[i] = (float) out[0];
        distances[i + 1] = (float) out[1];
    }

    if (rest_two == 1) { // it can only be 1 or 0
        size_t last = (height_src - 1);

        double agg = 0.0;
        double *current = source + last * width;
        for (size_t k = 0; k < width; ++k) {
            double f_0 = current[k] - input[k];
            f_0 = SQUARE(f_0);
            agg += f_0;
        }

        __m128d scalar_agg = _mm_set_sd(agg);
        __m128d sqrt_agg = _mm_sqrt_sd(scalar_agg, scalar_agg);

        double result = _mm_cvtsd_f64(sqrt_agg);

        distances[last] = (float) result;
    }
}
