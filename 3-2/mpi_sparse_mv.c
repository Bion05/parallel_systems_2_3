#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Generate random integer between min and max (inclusive)
int rand_int(int min, int max) {
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
void convert_to_csr(int *dense, int N, int **row_ptr, int **col_idx, int **values, int *nnz) {
    int count = 0;

    // Count non-zero elements in dense matrix
    for (int i = 0; i < N * N; i++) {
        if (dense[i] != 0) count++;
    }

    *nnz = count;

    // Allocate CSR arrays
    *row_ptr = malloc((N + 1) * sizeof(int));  // row pointer array
    *col_idx = malloc(count * sizeof(int));    // column indices array
    *values = malloc(count * sizeof(int));     // non-zero values array

    if (!*row_ptr || !*col_idx || !*values) {
        fprintf(stderr, "Failed to allocate CSR arrays\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int idx = 0;
    (*row_ptr)[0] = 0;


    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int val = dense[i * N + j];
            if (val != 0) {
                (*col_idx)[idx] = j;
                (*values)[idx] = val;
                idx++;
            }
        }
        (*row_ptr)[i + 1] = idx; // end of row i
    }
}

// CSR matrix-vector multiplication for rows [row_start, row_end)
void CSR_matvec(int *row_ptr, int *col_idx, int *values, int row_start, int row_end, int *x, int *y) {
    for (int i = row_start; i < row_end; i++) {
        int row_begin = row_ptr[i];       // start index of non-zero elements in row i
        int row_end_i = row_ptr[i + 1];   // end index (exclusive)
        int sum = 0;

        // Multiply non-zero elements of row i by corresponding x values
        for (int idx = row_begin; idx < row_end_i; idx++) {
            sum += values[idx] * x[col_idx[idx]];
        }
        y[i - row_start] = sum; // store result locally indexed
    }
}

// Dense matrix-vector multiplication for rows [row_start, row_end)
void dense_matvec(int *dense, int N, int row_start, int row_end, int *x, int *y) {
    for (int i = row_start; i < row_end; i++) {
        int sum = 0;

        // Multiply all elements in dense row i with vector x
        for (int j = 0; j < N; j++) {
            sum += dense[i * N + j] * x[j];
        }
        y[i - row_start] = sum; // store result locally indexed
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv); 

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Get current process rank
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Get total number of processes

    if (argc != 4) {
        if (rank == 0)
            fprintf(stderr, "Usage: %s <N> <sparsity> <repetitions>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int N = atoi(argv[1]);             // Matrix dimension
    double sparsity = atof(argv[2]);   // Sparsity ratio
    int repetitions = atoi(argv[3]);    // Number of repeated multiplications

    if (sparsity < 0.0 || sparsity > 1.0) {
    if (rank == 0) {
        fprintf(stderr, "Error: sparsity must be between 0.0 and 1.0\n");
    }
    MPI_Finalize();
    return 1;
    }

    int *dense_matrix = NULL;
    int *vector = NULL;
    int *result = NULL;

    int *row_ptr = NULL;
    int *col_idx = NULL;
    int *values = NULL;
    int nnz;

    double t_start, t_end;
    double csr_build_time = 0.0;
    double send_time = 0.0;
    double csr_compute_time = 0.0;
    double csr_total_time = 0.0;
    double dense_total_time = 0.0;

    if (rank == 0) {
        // Allocate dense matrix and vectors on root
        dense_matrix = malloc(N * N * sizeof(int));
        vector = malloc(N * sizeof(int));
        result = malloc(N * sizeof(int));

        srand(time(NULL)); // Seed random number generator

        // Create matrix and initialize vector with random values
        build_matrix(N, sparsity, dense_matrix);
        for (int i = 0; i < N; i++)
            vector[i] = rand_int(1, 10);

        // Convert dense matrix to CSR format and time it
        t_start = MPI_Wtime();
        convert_to_csr(dense_matrix, N, &row_ptr, &col_idx, &values, &nnz);
        t_end = MPI_Wtime();
        csr_build_time = t_end - t_start;
    }

    // Broadcast matrix size and non-zero count to all processes
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate CSR arrays and vectors on non-root processes
    if (rank != 0) {
        row_ptr = malloc((N + 1) * sizeof(int));
        col_idx = malloc(nnz * sizeof(int));
        values = malloc(nnz * sizeof(int));
        vector = malloc(N * sizeof(int));
        result = malloc(N * sizeof(int));
    }

    // Broadcast CSR matrix arrays and initial vector to all processes and time it
    t_start = MPI_Wtime();
    MPI_Bcast(row_ptr, N + 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(col_idx, nnz, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(values, nnz, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(vector, N, MPI_INT, 0, MPI_COMM_WORLD);
    t_end = MPI_Wtime();
    send_time = t_end - t_start;


    // Broadcast dense matrix to all for parallel dense matvec
    if (rank != 0) {
        dense_matrix = malloc(N * N * sizeof(int));
    }
    MPI_Bcast(dense_matrix, N * N, MPI_INT, 0, MPI_COMM_WORLD);


    // Calculate each process's assigned rows
    int rows_per_proc = N / size;
    int remainder = N % size;

    int row_start = rank * rows_per_proc + (rank < remainder ? rank : remainder);
    int row_end = row_start + rows_per_proc + (rank < remainder ? 1 : 0);
    int local_rows = row_end - row_start;

    // Allocate buffer for partial results computed locally
    int *local_result = malloc(local_rows * sizeof(int));

    // On root, prepare arrays for gathering partial results (recvcounts and displacements)
    int *recvcounts = NULL;
    int *displs = NULL;
    if (rank == 0) {
        recvcounts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
        int offset = 0;
        for (int i = 0; i < size; i++) {
            int start = i * rows_per_proc + (i < remainder ? i : remainder);
            int end = start + rows_per_proc + (i < remainder ? 1 : 0);
            recvcounts[i] = end - start;   // Number of rows for process i
            displs[i] = offset;            // Displacement in gathered array
            offset += recvcounts[i];
        }
    }

    // CSR sparse matrix-vector multiplication
    t_start = MPI_Wtime();
    for (int r = 0; r < repetitions; r++) {
        CSR_matvec(row_ptr, col_idx, values, row_start, row_end, vector, local_result);

        // Gather partial results from all processes to root
        MPI_Gatherv(local_result, local_rows, MPI_INT,
                    result, recvcounts, displs, MPI_INT,
                    0, MPI_COMM_WORLD);

        // Broadcast updated vector to all for next iteration
        MPI_Bcast(result, N, MPI_INT, 0, MPI_COMM_WORLD);

        // Update local vector copy on root
        if (rank == 0) {
            for (int i = 0; i < N; i++)
                vector[i] = result[i];
        }
    }
    t_end = MPI_Wtime();
    csr_compute_time = t_end - t_start;

    // Root prints timing summary for CSR operations
    if (rank == 0) {
        csr_total_time = csr_build_time + send_time + csr_compute_time;

        printf("CSR build time: %f seconds\n", csr_build_time);
        printf("Data send time: %f seconds\n", send_time);
        printf("CSR compute time: %f seconds\n", csr_compute_time);
        printf("Total CSR time: %f seconds\n", csr_total_time);
    }

    // Dense matrix-vector multiplication
    t_start = MPI_Wtime();

    for (int r = 0; r < repetitions; r++) {
        // Each process computes dense matvec on its assigned rows
        dense_matvec(dense_matrix, N, row_start, row_end, vector, local_result);

        // Gather partial results from all processes to root
        MPI_Gatherv(local_result, local_rows, MPI_INT,
                    result, recvcounts, displs, MPI_INT,
                    0, MPI_COMM_WORLD);

        // Broadcast updated vector to all for next iteration
        MPI_Bcast(result, N, MPI_INT, 0, MPI_COMM_WORLD);

        // Update local vector copy on root
        if (rank == 0) {
            for (int i = 0; i < N; i++)
                vector[i] = result[i];
        }
    }

    t_end = MPI_Wtime();
    dense_total_time = t_end - t_start;

    if (rank == 0) {
        printf("Dense total time: %f seconds\n", dense_total_time);
    }

    // Free dynamically allocated memory
    free(row_ptr);
    free(col_idx);
    free(values);
    free(vector);
    free(result);
    free(local_result);
    free(dense_matrix);

    if (rank == 0) {
        free(recvcounts);
        free(displs);
    }

    MPI_Finalize();
    return 0;
}
