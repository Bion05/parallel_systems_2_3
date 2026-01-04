#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>

#define THRESHOLD 10000 //used to reduce unnecessary task 

//merges 2 subarrays of A (a[l..m] + a[m+1..r])
void merge(int *A, int l, int m, int r) {
    int i = l, j = m + 1, k = 0;
    int *temp = (int *)malloc((r - l + 1) * sizeof(int));

    //copy data to temp sorted
    while (i <= m && j <= r) 
        temp[k++] = (A[i] <= A[j]) ? A[i++] : A[j++];
    
    //copy remaining data of the subarray that hasn't finished yet
    while (i <= m) temp[k++] = A[i++];
    while (j <= r) temp[k++] = A[j++];

    //copy data to A
    memcpy(A + l, temp, (r - l + 1) * sizeof(int));
    free(temp);
}

//serial mergesort
void mergesort_serial(int *A, int l, int r) {
    if (l < r) {
        int m = (l + r) / 2;
        mergesort_serial(A, l, m);
        mergesort_serial(A, m + 1, r);
        merge(A, l, m, r);
    }
}

//parallel mergesort
void mergesort_parallel(int *A, int l, int r) {
    if (l < r) {
        int m = (l + r) / 2;

        #pragma omp task if (r - l > THRESHOLD)
        mergesort_parallel(A, l, m);

        #pragma omp task if (r - l > THRESHOLD)
        mergesort_parallel(A, m + 1, r);

        #pragma omp taskwait
        merge(A, l, m, r);
    }
}

//checks if A is sorted
int is_sorted(int *A, int n) {
    for (int i = 1; i < n; i++)
        if (A[i - 1] > A[i]) return 0;
    return 1;
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Usage: %s <size> <\"serial\"/\"parallel\"> <thread number>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    int threads = atoi(argv[3]);
    int *A = (int *)malloc(n * sizeof(int));

    srand(42); // deterministic randomness
    for (int i = 0; i < n; i++)
        A[i] = rand();

    double start, end;

    //serial sort
    if (strcmp(argv[2], "serial") == 0) {
        start = omp_get_wtime();
        mergesort_serial(A, 0, n - 1);
        end = omp_get_wtime();
    } else if (strcmp(argv[2], "parallel") == 0) { //parallel sort
        omp_set_num_threads(threads);
        start = omp_get_wtime();
        #pragma omp parallel
        {
            #pragma omp single
            mergesort_parallel(A, 0, n - 1);
        }
        end = omp_get_wtime();
    } else {
        printf("Usage: %s <size> <\"serial\"/\"parallel\"> <thread number>\n", argv[0]);
        return 1;
    }

    printf("Sorted: %s\n", is_sorted(A, n) ? "YES" : "NO");
    printf("Time: %f seconds\n", end - start);

    free(A);
    return 0;
}
