#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

// Generate random integer between min and max (inclusive)
int rand_int(int min, int max){
    return min + rand() % (max - min + 1);
}

// Create a random sparse matrix with given sparsity
void build_matrix(int N, double sparsity, int *dense_matrix) {
    int total = N * N;
    int zero_count = (int)(total * sparsity);

    // Fill matrix with random values between 1 and 10
    for (int i = 0; i < total; i++)
        dense_matrix[i] = rand_int(1, 10);

    // Randomly set some elements to zero to achieve desired sparsity
    for (int i = 0; i < zero_count; i++) {
        int pos;
        do {
            pos = rand_int(0, total - 1);
        } while (dense_matrix[pos] == 0);
        dense_matrix[pos] = 0;
    }
}

// Convert dense matrix to CSR format outputs
void convert_to_csr(int *dense, int N, int **row_ptr, int **col_idx, int **values, int *nnz, int num_threads){
    int *row_nnz = malloc(N * sizeof(int));
    if(!row_nnz){
        fprintf(stderr, "Allocation faied\n");
        exit(1);
    }

    // Count non zero elements per row (parallel)
    #pragma omp parallel for num_threads(num_threads)
    for(int i = 0; i < N; i++){
        int count = 0;
        for(int j = 0; j < N; j++){
            if(dense[i*N + j] != 0) count++;
        }
        row_nnz[i] = count;
    }

    // Total NNZ
    int total_nnz = 0;
    for(int i = 0; i < N; i++){
        total_nnz += row_nnz[i];
    }
    *nnz = total_nnz;

    // Allocate CSR arrays
    *row_ptr = malloc((N + 1) * sizeof(int));
    *col_idx = malloc(total_nnz * sizeof(int));
    *values = malloc(total_nnz * sizeof(int));
    if(!*row_ptr || !*col_idx || !*values){
        fprintf(stderr, "CSR allocaion failed\n");
        exit(1);
    }

    // Compute row_ptr prefix sum
    (*row_ptr)[0] = 0;
    for(int i = 0; i < N; i++){
        (*row_ptr)[i+1] = (*row_ptr)[i] + row_nnz[i];
    }

    // Fill values and col_idx arrays in parallel
    #pragma omp parallel for num_threads(num_threads)
    for(int i = 0; i < N; i++){
        int idx = (*row_ptr)[i];
        for(int j = 0; j < N; j++){
            int val = dense[i*N + j];
            if(val != 0){
                (*values)[idx] = val;
                (*col_idx)[idx] = j;
                idx++;
            }
        }
    }
    free(row_nnz);
}

// CSR matrix-vector multiplication (parallel)
void CSR_matvec(int *row_ptr, int *col_idx, int *values, int N, int *x, int *y, int num_threads){
    #pragma omp parallel for num_threads(num_threads)
    for(int i = 0; i < N; i++){
        int sum = 0;
        for(int j = row_ptr[i]; j < row_ptr[i+1]; j++){
            sum += values[j] * x[col_idx[j]];
        }
        y[i] = sum;
    }
}

// Dense matrix-vector multiplication (parallel)
void dense_matvec(int *dense, int N, int *x, int *y, int num_threads){
    #pragma omp parallel for num_threads(num_threads)
    for(int i = 0; i < N; i++){
        int sum = 0;
        for(int j = 0; j < N; j++){
            sum += dense[i*N + j] * x[j];
        }
        y[i] = sum;
    }
}

int check_results(int *a, int *b, int N) {
    for (int i = 0; i < N; i++) {
        if (a[i] != b[i])
            return 0;
    }
    return 1;
}


int main(int argc, char *argv[]){
    if(argc != 5){
        fprintf(stderr, "Usage: %s <N> <sparsity> <repetitions> <num_threads>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    double sparsity = atof(argv[2]);
    int repetitions = atoi(argv[3]);
    int num_threads = atoi(argv[4]);

    if(sparsity < 0.0 || sparsity > 1.0){
        fprintf(stderr, "Error: sparsity must be between 0.0 and 1.0\n");
        return 1;
    }

    srand(time(NULL));

    int *dense_matrix = malloc(N * N * sizeof(int));
    int *vector = malloc(N * sizeof(int));
    int *y = malloc(N * sizeof(int));

    if(!dense_matrix || !vector || !y){
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    // Generate random dense matrix and vector (not timed)
    build_matrix(N, sparsity, dense_matrix);
    for(int i = 0; i < N; i++){
        vector[i] = rand_int(1, 10);
    }

    int *row_ptr, *col_idx, *values;
    int nnz;

    // Construct CSR representation and time it
    double t_start = omp_get_wtime();
    convert_to_csr(dense_matrix, N, &row_ptr, &col_idx, &values, &nnz, num_threads);
    double t_end = omp_get_wtime();
    printf("CSR build time: %f seconds\n", t_end - t_start);

    // CORRECTNESS CHECK (not timed)
    int *y_csr   = malloc(N * sizeof(int));
    int *y_dense = malloc(N * sizeof(int));

    if (!y_csr || !y_dense) {
        fprintf(stderr, "Allocation failed in correctness check\n");
        exit(1);
    }

    // One multiplication only
    CSR_matvec(row_ptr, col_idx, values, N, vector, y_csr, num_threads);
    dense_matvec(dense_matrix, N, vector, y_dense, num_threads);

    if (!check_results(y_csr, y_dense, N)) {
        printf("ERROR: CSR and Dense results differ!\n");
    } else {
        printf("Correctness check passed ✔️\n");
    }

    free(y_csr);
    free(y_dense);

    // CSR repeated multiplication
    int *x = malloc(N * sizeof(int));
    for(int i = 0; i < N; i++){
        x[i] = vector[i];
    }

    t_start = omp_get_wtime();
    for(int i = 0; i < repetitions; i++){
        CSR_matvec(row_ptr, col_idx, values, N, x, y, num_threads);
        int *tmp = x; x = y; y = tmp;
    }
    t_end = omp_get_wtime();
    printf("CSR repeated multiplication time: %f seconds\n", t_end - t_start);

    // Dense repeated multiplications
    for(int i = 0; i < N; i++){
        x[i] = vector[i];
    }
    t_start = omp_get_wtime();
    for(int j = 0; j < repetitions; j++){
        dense_matvec(dense_matrix, N, x, y, num_threads);
        int *tmp = x; x = y; y = tmp;
    }
    t_end = omp_get_wtime();
    printf("Dense repeated multiplication time: %f seconds\n", t_end - t_start);

    // Free memory
    free(dense_matrix);
    free(vector);
    free(y);
    free(x);
    free(col_idx);
    free(row_ptr);
    free(values);

    return 0;
}