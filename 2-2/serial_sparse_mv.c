#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* ----------------- Utility ----------------- */

int rand_int(int min, int max) {
    return min + rand() % (max - min + 1);
}

/* ----------------- Matrix generation ----------------- */

void build_matrix(int N, double sparsity, int *dense_matrix) {
    for (int i = 0; i < N * N; i++) {
        double r = rand() / (double) RAND_MAX;
        if (r < sparsity)
            dense_matrix[i] = 0;
        else
            dense_matrix[i] = rand_int(1, 10);
    }
}

/* ----------------- Dense matvec (serial) ----------------- */

void dense_matvec_serial(int *dense, int N, int *x, int *y) {
    for (int i = 0; i < N; i++) {
        int sum = 0;
        for (int j = 0; j < N; j++) {
            sum += dense[i * N + j] * x[j];
        }
        y[i] = sum;
    }
}

/* ----------------- CSR construction (serial) ----------------- */

void convert_to_csr_serial(int *dense, int N,
                           int **row_ptr, int **col_idx,
                           int **values, int *nnz) {

    int *row_nnz = malloc(N * sizeof(int));
    if (!row_nnz) {
        fprintf(stderr, "Allocation failed\n");
        exit(1);
    }

    for (int i = 0; i < N; i++) {
        int count = 0;
        for (int j = 0; j < N; j++) {
            if (dense[i * N + j] != 0)
                count++;
        }
        row_nnz[i] = count;
    }

    *row_ptr = malloc((N + 1) * sizeof(int));
    (*row_ptr)[0] = 0;
    for (int i = 0; i < N; i++)
        (*row_ptr)[i + 1] = (*row_ptr)[i] + row_nnz[i];

    *nnz = (*row_ptr)[N];

    *col_idx = malloc(*nnz * sizeof(int));
    *values  = malloc(*nnz * sizeof(int));

    if (!*col_idx || !*values) {
        fprintf(stderr, "CSR allocation failed\n");
        exit(1);
    }

    for (int i = 0; i < N; i++) {
        int idx = (*row_ptr)[i];
        for (int j = 0; j < N; j++) {
            int val = dense[i * N + j];
            if (val != 0) {
                (*values)[idx] = val;
                (*col_idx)[idx] = j;
                idx++;
            }
        }
    }

    free(row_nnz);
}

/* ----------------- CSR matvec (serial) ----------------- */

void CSR_matvec_serial(int *row_ptr, int *col_idx,
                       int *values, int N,
                       int *x, int *y) {

    for (int i = 0; i < N; i++) {
        int sum = 0;
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
            sum += values[j] * x[col_idx[j]];
        }
        y[i] = sum;
    }
}

/* ----------------- Main ----------------- */

int main(int argc, char *argv[]) {

    if (argc != 4) {
        fprintf(stderr,
            "Usage: %s <N> <sparsity> <repetitions>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    double sparsity = atof(argv[2]);
    int repetitions = atoi(argv[3]);

    srand(time(NULL));

    int *dense_matrix = malloc(N * N * sizeof(int));
    int *vector = malloc(N * sizeof(int));
    int *x = malloc(N * sizeof(int));
    int *y = malloc(N * sizeof(int));

    if (!dense_matrix || !vector || !x || !y) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    build_matrix(N, sparsity, dense_matrix);

    for (int i = 0; i < N; i++)
        vector[i] = rand_int(1, 10);

    /* --------- CSR construction timing --------- */

    int *row_ptr, *col_idx, *values;
    int nnz;

    clock_t t1 = clock();
    convert_to_csr_serial(dense_matrix, N,
                          &row_ptr, &col_idx,
                          &values, &nnz);
    clock_t t2 = clock();

    printf("Serial CSR build time: %f seconds\n",
           (double)(t2 - t1) / CLOCKS_PER_SEC);

    /* --------- CSR repeated multiplication --------- */

    for (int i = 0; i < N; i++)
        x[i] = vector[i];

    t1 = clock();
    for (int r = 0; r < repetitions; r++) {
        CSR_matvec_serial(row_ptr, col_idx, values, N, x, y);
        int *tmp = x; x = y; y = tmp;
    }
    t2 = clock();

    printf("Serial CSR repeated multiplication time: %f seconds\n",
           (double)(t2 - t1) / CLOCKS_PER_SEC);

    /* --------- Dense repeated multiplication --------- */

    for (int i = 0; i < N; i++)
        x[i] = vector[i];

    t1 = clock();
    for (int r = 0; r < repetitions; r++) {
        dense_matvec_serial(dense_matrix, N, x, y);
        int *tmp = x; x = y; y = tmp;
    }
    t2 = clock();

    printf("Serial Dense repeated multiplication time: %f seconds\n",
           (double)(t2 - t1) / CLOCKS_PER_SEC);

    /* --------- Cleanup --------- */

    free(dense_matrix);
    free(vector);
    free(x);
    free(y);
    free(row_ptr);
    free(col_idx);
    free(values);

    return 0;
}