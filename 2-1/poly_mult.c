#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>


//Generates a random complete polynomial
void rand_polly(int *poly, int n) {
    for (int i = 0; i <= n; ++i) {
        int coeff = 0;
        // coeff range is [-10,10] - {0} 
        while (coeff == 0) {
            coeff = (rand() % 21) - 10; //[-x,x] - {0} Range => [(rand()%(2x+1))-1]
        }
        poly[i] = coeff;
    }
}

//Serial multiplication
void mult_serial(int *A, int *B, int *C, int n) {
    for (int i = 0; i <= 2*n; i++)
        C[i] = 0;

    for (int i = 0; i <= n; i++) {
        for (int j = 0; j <= n; j++) {
            C[i+j] += A[i] * B[j];
        }
    }
}

//Parallel multiplication using OpenMP
void mult_openmp(int *A, int *B, int *C, int n, int thread_count) {
    int total_coeffs = 2 * n + 1;

    #pragma omp parallel for num_threads(thread_count) schedule(static)
    for (int k = 0; k < total_coeffs; k++) {
        int sum = 0;
        for (int i = 0; i <= n; i++) {
            int j = k - i;
            if (j >= 0 && j <= n) { //ignore out of bounds j
                sum += A[i] * B[j];
            }
        }
        C[k] = sum;
    }
}

//compares 2 arrays (polinomials)
int ar_compare(int* A, int *B,int degree){
    for(int i = 0; i<=degree; i++)if(A[i]!=B[i])return 0;
    return 1;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: %s <degree n> <thread_count>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    int thread_count = atoi(argv[2]);

    if (n < 0 || thread_count <= 0) {
        printf("Error: n must be >= 0 and thread_count > 0\n");
        return 1;
    }

    srand(time(NULL)); // Seed RNG

    int *A = malloc((n+1) * sizeof(int));
    int *B = malloc((n+1) * sizeof(int));
    int *Serial = malloc((2*n+1) * sizeof(int));
    int *OpenMP = malloc((2*n+1) * sizeof(int));

    double start, end;

    start = omp_get_wtime();
    rand_polly(A, n);
    rand_polly(B, n);
    end = omp_get_wtime();
    printf("Initialize time: %f seconds\n", end - start);

    start = omp_get_wtime();
    mult_serial(A, B, Serial, n);
    end = omp_get_wtime();
    printf("Serial multiplication time: %f seconds\n", end - start);

    start = omp_get_wtime();
    mult_openmp(A, B, OpenMP, n, thread_count);
    end = omp_get_wtime();
    printf("Parallel multiplication time: %f seconds\n", end - start);

    if (ar_compare(Serial, OpenMP, 2*n))
        printf("The results match\n");
    else
        printf("The results do not match\n");

    free(A);
    free(B);
    free(Serial);
    free(OpenMP);

    return 0;
}
