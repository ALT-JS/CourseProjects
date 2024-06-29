#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <immintrin.h>

// void write_array_to_file(const char *filename, double *array, int rows, int cols, const char *title) {
//   FILE *file = fopen(filename, "a");

//   fprintf(file, "%s\n", title);
//   for (int i = 0; i < rows; i++) {
//     for (int j = 0; j < cols; j++) {
//       fprintf(file, "%f\t", array[i * cols + j]);
//     }
//     fprintf(file, "\n");
//   }
//   fprintf(file, "\n\n\n");

//   fclose(file);
// }

void impl_experiment(int N, int step, double *p) {
    if (N & 1) {
        for (int k = 0; k < step; k++) {
            #pragma omp parallel for
            for (int i = 1; i < N - 1; i++) {
                for (int j = 2 - ((k & 1) ^ (i & 1)); j < N - 1; j += 2) {
                    p[i * N + j] = (p[(i - 1) * N + j] + p[(i + 1) * N + j] +
                                        p[i * N + j + 1] + p[i * N + j - 1]) /
                                        4.0f;
                }
            }
        }
    } else {
        int Nhalf = N >> 1;
        double *A = (double *)aligned_alloc(32, N * Nhalf * sizeof(double));
        double *B = (double *)aligned_alloc(32, N * Nhalf * sizeof(double));
        // double *A = calloc(N * N / 2, sizeof(double));
        // double *B = calloc(N * N / 2, sizeof(double));
        omp_set_num_threads(14);
        #pragma omp parallel for
        for (int i = 0; i < N; i++) { 
            for (int j = (i & 1); j < N - 1 + (i & 1); j += 2) {
                A[i * Nhalf + j / 2] = p[i * N + j];
            }
        }
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            for (int j = 1 - (i & 1); j < N - (i & 1); j += 2) {
                B[i * Nhalf + j / 2] = p[i * N + j];
            }
        }
        // write_array_to_file("debug.txt", A, N, N/2, "A");
        // write_array_to_file("debug.txt", B, N, N/2, "B");
        // write_array_to_file("debug.txt", p, N, N, "P");

        // int *index_mat = (int *)calloc(N * N, sizeof(int));
        int k;
        for (k = 0; k < step - (step & 1);) {
            // if (k & 1 == 0) {
                #pragma omp parallel for
                for (int i = 1; i < N - 1; i++) {
                    int j;
                    for (j = !((k & 1) ^ (i & 1)); j < (Nhalf) - 1 + !((k & 1) ^ (i & 1)) - 3; j += 4) {
                        __m256d t = _mm256_loadu_pd(&B[(i - 1) * Nhalf + j]);
                        __m256d b = _mm256_loadu_pd(&B[(i + 1) * Nhalf + j]);
                        __m256d l = _mm256_loadu_pd(&B[i * Nhalf + j]);
                        __m256d r = _mm256_loadu_pd(&B[i * Nhalf + j + 1 - 2 * !((k & 1) ^ (i & 1))]);

                        __m256d result = _mm256_add_pd(t, b);
                        result = _mm256_add_pd(result, l);
                        result = _mm256_add_pd(result, r);
                        result = _mm256_mul_pd(result, _mm256_set1_pd(0.25));
                        _mm256_storeu_pd(&A[i * Nhalf + j], result);
                        
                    }
                    for (; j < (Nhalf) - 1 + !((k & 1) ^ (i & 1)); j++) {
                        A[i * Nhalf + j] = (B[(i - 1) * Nhalf + j] + B[i * Nhalf + j] + B[(i + 1) * Nhalf + j] + B[i * Nhalf + j + 1 - 2 * !((k & 1) ^ (i & 1))]) * 0.25f;
                    }
                }
            // } else {
                k = k + 1;
                #pragma omp parallel for
                for (int i = 1; i < N - 1; i++) {
                    int j;
                    for (j = !((k & 1) ^ (i & 1)); j < (Nhalf) - 1 + !((k & 1) ^ (i & 1)) - 3; j += 4) {
                        __m256d t = _mm256_loadu_pd(&A[(i - 1) * Nhalf + j]);
                        __m256d b = _mm256_loadu_pd(&A[(i + 1) * Nhalf + j]);
                        __m256d l = _mm256_loadu_pd(&A[i * Nhalf + j]);
                        __m256d r = _mm256_loadu_pd(&A[i * Nhalf + j + 1 - 2 * !((k & 1) ^ (i & 1))]);

                        __m256d result = _mm256_add_pd(t, b);
                        result = _mm256_add_pd(result, l);
                        result = _mm256_add_pd(result, r);
                        result = _mm256_mul_pd(result, _mm256_set1_pd(0.25));
                        _mm256_storeu_pd(&B[i * Nhalf + j], result);
                        
                    }
                    for (; j < (Nhalf) - 1 + !((k & 1) ^ (i & 1)); j++) {
                        B[i * Nhalf + j] = (A[(i - 1) * Nhalf + j] + A[i * Nhalf + j] + A[(i + 1) * Nhalf + j] + A[i * Nhalf + j + 1 - 2 * !((k & 1) ^ (i & 1))]) * 0.25f;
                    }
                }
                k = k + 1;
            // }
        }
        for (; k < step; k++) {
            #pragma omp parallel for
            for (int i = 1; i < N - 1; i++) {
                int j;
                for (j = !((k & 1) ^ (i & 1)); j < (Nhalf) - 1 + !((k & 1) ^ (i & 1)) - 3; j += 4) {
                    __m256d t = _mm256_loadu_pd(&B[(i - 1) * Nhalf + j]);
                    __m256d b = _mm256_loadu_pd(&B[(i + 1) * Nhalf + j]);
                    __m256d l = _mm256_loadu_pd(&B[i * Nhalf + j]);
                    __m256d r = _mm256_loadu_pd(&B[i * Nhalf + j + 1 - 2 * !((k & 1) ^ (i & 1))]);

                    __m256d result = _mm256_add_pd(t, b);
                    result = _mm256_add_pd(result, l);
                    result = _mm256_add_pd(result, r);
                    result = _mm256_mul_pd(result, _mm256_set1_pd(0.25));
                    _mm256_storeu_pd(&A[i * Nhalf + j], result);
                    
                }
                for (; j < (Nhalf) - 1 + !((k & 1) ^ (i & 1)); j++) {
                    A[i * Nhalf + j] = (B[(i - 1) * Nhalf + j] + B[i * Nhalf + j] + B[(i + 1) * Nhalf + j] + B[i * Nhalf + j + 1 - 2 * !((k & 1) ^ (i & 1))]) * 0.25f;
                }
            }
        }

        #pragma omp parallel for
        for (int i = 0; i < N; i++) { 
            for (int j = (i & 1); j < N - 1 + (i & 1); j += 2) {
                p[i * N + j] = A[i * Nhalf + j / 2];
            }
        }
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            for (int j = 1 - (i & 1); j < N - (i & 1); j += 2) {
                p[i * N + j] = B[i * Nhalf + j / 2];
            }
        }
        
        free(A);
        free(B);
    }
}