#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

// Generates a random complete polynomial of degree n
void rand_poly(int *poly, int n){
    for(int i = 0; i <= n; i++){
        int c = 0;
        while(c == 0){
            // All coefficients are integers in the range [-10, 10].
            c = (rand() % 21) - 10;
        }
        poly[i] = c;
    }
}

int main(int argc, char *argv[]){
    int rank, size;
    int n;

    // Initialize MPI environment
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if(argc != 2){
        if(rank == 0){
            printf("Usage: %s <degree n>\n", argv[0]);
        }
        MPI_Finalize();
        return 0;
    }

    n = atoi(argv[1]);
    int result_size = 2*n + 1; // Degree of the result polynomial

    int *A = NULL;  // First polynomial
    int *B = NULL;  // Second polynomial
    int *C = NULL;  // Result polynomial (only used by rank 0)

    double t_total_start, t_total_end;
    double t_send = 0.0, t_compute = 0.0, t_recv = 0.0;

    // Rank 0 initializes the input polynomials
    if(rank == 0){
        srand(time(NULL));
        A = malloc((n+1)*sizeof(int));
        B = malloc((n+1)*sizeof(int));
        C = calloc(result_size, sizeof(int));

        rand_poly(A, n);
        rand_poly(B, n);
    }

    // Synchronize all processes before timing
    MPI_Barrier(MPI_COMM_WORLD);
    t_total_start = MPI_Wtime();
    
    // Measure data distribution time
    double t1 = MPI_Wtime();  

    if(rank != 0){
        A = malloc((n+1)*sizeof(int));
        B = malloc((n+1)*sizeof(int));
    }

    // Broadcast polynomials A and B to all processes
    MPI_Bcast(A, n+1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, n+1, MPI_INT, 0, MPI_COMM_WORLD);

    t_send = MPI_Wtime() - t1;

    // Parallel computation phase
    double t2 = MPI_Wtime();
    int coeffs_per_proc = result_size / size;
    int remainder = result_size % size;

    // Compute the range of coefficients handled by this process
    int start = rank * coeffs_per_proc + (rank < remainder ? rank : remainder);
    int local_n = coeffs_per_proc + (rank < remainder ? 1 : 0);

    int *local_C = calloc(local_n, sizeof(int));

    // Compute assigned coefficients of the result polynomial
    for(int i = 0; i < local_n; i++){
        int global_i = start + i;
        int sum = 0;
        for(int j = 0; j <= n; j++){
            int k = global_i - j;
            if(k >= 0 && k <= n){
                sum += A[j] * B[k];
            }
        }
        local_C[i] = sum;
    }

    t_compute = MPI_Wtime() - t2;

    // Gather partial results at rank 0
    double t3 = MPI_Wtime();

    int *recv_counts = NULL;
    int *displs = NULL;

    if(rank == 0){
        recv_counts = malloc(size*sizeof(int));
        displs = malloc(size*sizeof(int));
        int offset = 0;
        for(int i = 0; i < size; i++){
            recv_counts[i] = coeffs_per_proc + (i < remainder ? 1 : 0);
            displs[i] = offset;
            offset += recv_counts[i];
        }
    }

    MPI_Gatherv(local_C, local_n, MPI_INT, C, 
                recv_counts, displs, MPI_INT, 
                0, MPI_COMM_WORLD);

    t_recv = MPI_Wtime() - t3;
    t_total_end = MPI_Wtime();

    // Print timing results (only by rank 0)
    if(rank == 0){
        printf("Send time:      %f seconds\n", t_send);
        printf("Compute time:   %f seconds\n", t_compute);
        printf("Receive time:   %f seconds\n", t_recv);
        printf("Total time:     %f seconds\n", t_total_end - t_total_start);
    }

    // Free allocated memory
    free(A);
    free(B);
    free(local_C);

    if(rank == 0){
        free(C);
        free(recv_counts);
        free(displs);
    }

    // Finalize MPI environment
    MPI_Finalize();
    return 0;
}