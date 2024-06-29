#include <immintrin.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

void impl_old(int N, int step, double *p) {
  // float *fp = NULL;
  float *fp = (float *)aligned_alloc(32, N * N * sizeof(float));
  int fp_judge = 0;
  // __attribute_maybe_unused__ int retvalfp = posix_memalign((void**)&fp, 32, N * N * sizeof(float));

  #pragma omp parallel for
  for (int i = 0; i < N * N; i++) {
    if (p[i] >= 64.0) {
      #pragma omp critical
      fp_judge = 1;
    } else {
      fp[i] = (float)p[i];
    }
  }
  if (fp_judge) {
    // double *p_next = NULL;
    // __attribute_maybe_unused__ int retval = posix_memalign((void**)&p_next, 32, N * N * sizeof(double));
    double *p_next = (double *)aligned_alloc(32, N * N * sizeof(double));
    memcpy(p_next, p, N * N * sizeof(double));
    if (step % 2 == 1) {
      step--;
    }
    #pragma omp parallel
    {
      for (int k = 0; k < step; k++) {
        #pragma omp for collapse(2)
        for (int i = 1; i < N - 1; i++) {
          for (int j = 1; j < N - 1; j += 4) {
            if (j + 3 < N - 1) {
              __m256d t = _mm256_loadu_pd(&p[(i - 1) * N + j]);
              __m256d b = _mm256_loadu_pd(&p[(i + 1) * N + j]);
              __m256d l = _mm256_loadu_pd(&p[i * N + j - 1]);
              __m256d r = _mm256_loadu_pd(&p[i * N + j + 1]);

              __m256d result = _mm256_add_pd(t, b);
              result = _mm256_add_pd(result, l);
              result = _mm256_add_pd(result, r);
              result = _mm256_mul_pd(result, _mm256_set1_pd(0.25));
              _mm256_storeu_pd(&p_next[i * N + j], result);
            } else {
              for (int m = j; m < N - 1; m++) {
                p_next[i * N + m] = (p[(i - 1) * N + m] + p[(i + 1) * N + m] +
                                    p[i * N + m + 1] + p[i * N + m - 1]) /
                                    4.0f;
              }
            }
            
          }
        }

        #pragma omp barrier // sync threads
        #pragma omp single //  before swap
        {
          double *temp = p;
          p = p_next;
          p_next = temp;
        }
      }
    }
    // if (step % 2 == 1) {
    //   memcpy(p_next, p, N * N * sizeof(double));
    // }
    free(p_next);
    free(fp);
  } else {
    // float *fp_next = NULL;
    // __attribute_maybe_unused__ int retvalfpn = posix_memalign((void**)&fp_next, 32, N * N * sizeof(float));
    float *fp_next = (float *)aligned_alloc(32, N * N * sizeof(float));
    memcpy(fp_next, fp, N * N * sizeof(float));
    if (step % 2 == 1) {
      step--;
    }
    #pragma omp parallel
    {
      for (int k = 0; k < step; k++) {
        #pragma omp for collapse(2)
        for (int i = 1; i < N - 1; i++) {
          for (int j = 1; j < N - 1; j += 8) {
            if (j + 7 < N - 1) {
              __m256 t = _mm256_loadu_ps(&fp[(i - 1) * N + j]);
              __m256 b = _mm256_loadu_ps(&fp[(i + 1) * N + j]);
              __m256 l = _mm256_loadu_ps(&fp[i * N + j - 1]);
              __m256 r = _mm256_loadu_ps(&fp[i * N + j + 1]);

              __m256 result = _mm256_add_ps(t, b);
              result = _mm256_add_ps(result, l);
              result = _mm256_add_ps(result, r);
              result = _mm256_mul_ps(result, _mm256_set1_ps(0.25f));
              _mm256_storeu_ps(&fp_next[i * N + j], result);
            } else {
              for (int m = j; m < N - 1; m++) {
                fp_next[i * N + m] = (fp[(i - 1) * N + m] + fp[(i + 1) * N + m] +
                                    fp[i * N + m + 1] + fp[i * N + m - 1]) / 4.0f;
              }
            }
            
          }
        }

        #pragma omp barrier // sync threads
        #pragma omp single //  before swap
        {
          float *temp = fp;
          fp = fp_next;
          fp_next = temp;
        }
      }

      // if (step % 2 == 1) {
      //   // memcpy(p_next, p, N * N * sizeof(double));
      //   #pragma omp for collapse(2)
      //   for (int i = 1; i < N - 1; i++) {
      //     for (int j = 1; j < N - 1; j += 8) {
      //       if (j + 7 < N - 1) {
      //         __m256 t = _mm256_loadu_ps(&p[(i - 1) * N + j]);
      //         __m256 b = _mm256_loadu_ps(&p[(i + 1) * N + j]);
      //         __m256 l = _mm256_loadu_ps(&p[i * N + j - 1]);
      //         __m256 r = _mm256_loadu_ps(&p[i * N + j + 1]);

      //         __m256 result = _mm256_add_ps(t, b);
      //         result = _mm256_add_ps(result, l);
      //         result = _mm256_add_ps(result, r);
      //         result = _mm256_mul_ps(result, _mm256_set1_ps(0.25f));
      //         _mm256_storeu_ps(&p_next[i * N + j], result);
      //       } else {
      //         for (int m = j; m < N - 1; m++) {
      //           p_next[i * N + m] = (p[(i - 1) * N + m] + p[(i + 1) * N + m] +
      //                               p[i * N + m + 1] + p[i * N + m - 1]) /
      //                               4.0f;
      //         }
      //       }
      //     }
      //   }
      //   #pragma omp barrier // sync threads
      //   #pragma omp single //  before swap
      //   {
      //     float *temp = p;
      //     p = p_next;
      //     p_next = temp;
      //   }
      // }
    }

    #pragma omp parallel for
    for (int i = 0; i < N * N; i++) {
      p[i] = (double)fp[i];
    }
    free(fp);
    free(fp_next);
  }
}

void impl_block_idx_old(int N, int step, double *p) {
    double *p_next = (double *)malloc(N * N * sizeof(double));
    memcpy(p_next, p, N * N * sizeof(double));
    int *index_mat = (int *)calloc(N * N, sizeof(int));
    if (step % 2 == 1) {
      step--;
    }
    // for (int i = 0; i < N * N; i++) {
    //     index_mat[i] = 1;
    // }

    int B = 54;
    
    for (int k = 0; k < step; k++) {
        #pragma omp parallel for collapse(2)
        for (int bi = 1; bi < N - 1; bi += B) {
            for (int bj = 1; bj < N - 1; bj += B) {
                for (int i = bi; i < bi + B && i < N - 1; i++) {
                    for (int j = bj; j < bj + B && j < N - 1; j++) {
                        if (!index_mat[(i - 1) * N + j] || !index_mat[(i + 1) * N + j] || !index_mat[i * N + j + 1] || !index_mat[i * N + j - 1]) {
                            p_next[i * N + j] = (p[(i - 1) * N + j] + p[(i + 1) * N + j] + p[i * N + j + 1] + p[i * N + j - 1]) / 4.0f;
                            if (p_next[i * N + j] != p[i * N + j]) {
                                index_mat[i * N + j] = 0;
                            } else {
                                index_mat[i * N + j] = 1;
                            }
                        } else {
                            index_mat[i * N + j] = 1;
                        }
                        // p_next[i * N + j] = (p[(i - 1) * N + j] + p[(i + 1) * N + j] + p[i * N + j + 1] + p[i * N + j - 1]) / 4.0f;
                    }
                }
            }
        }

        #pragma omp single
        {
            double *temp = p;
            p = p_next;
            p_next = temp;
        }
    }
    free(p_next);
    free(index_mat);
}

// void impl(int N, int step, double *p) {
//     if (N % 2 == 1) {
//         for (int k = 0; k < step; k++) {
//             #pragma omp parallel for
//             for (int i = 1; i < N - 1; i++) {
//                 for (int j = 2 - ((k % 2) ^ (i % 2)); j < N - 1; j += 2) {
//                     p[i * N + j] = (p[(i - 1) * N + j] + p[(i + 1) * N + j] +
//                                         p[i * N + j + 1] + p[i * N + j - 1]) /
//                                         4.0f;
//                 }
//             }
//         }
//     } else {
//         double *A = (double *)malloc((N/2) * N * sizeof(double));
//         double *B = (double *)malloc((N/2) * N * sizeof(double));
        
//         omp_set_num_threads(14);
//         #pragma omp parallel for
//         for (int i = 0; i < N; i++) { 
//             for (int j = (i % 2); j < N - 1 + (i % 2); j += 2) {
//                 A[i * N / 2 + j / 2] = p[i * N + j];
//             }
//         }
//         #pragma omp parallel for
//         for (int i = 0; i < N; i++) {
//             for (int j = 1 - (i % 2); j < N - (i % 2); j += 2) {
//                 B[i * N / 2 + j / 2] = p[i * N + j];
//             }
//         }
//         // write_array_to_file("debug.txt", A, N, N/2, "A");
//         // write_array_to_file("debug.txt", B, N, N/2, "B");
//         // write_array_to_file("debug.txt", p, N, N, "P");

//         // int *index_mat = (int *)calloc(N * N, sizeof(int));
//         for (int k = 0; k < step; k++) {
//             if (k % 2 == 0) {
//                 #pragma omp parallel for
//                 for (int i = 1; i < N - 1; i++) {
//                     for (int j = !((k % 2) ^ (i % 2)); j < (N / 2) - 1 + !((k % 2) ^ (i % 2)); j++) {
//                         A[i * N / 2 + j] = (B[(i - 1) * N / 2 + j] + B[i * N / 2 + j] + B[(i + 1) * N / 2 + j] + B[i * N / 2 + j + 1 - 2 * !((k % 2) ^ (i % 2))]) * 0.25f;
//                     }
//                 }
//             } else {
//                 #pragma omp parallel for
//                 for (int i = 1; i < N - 1; i++) {
//                     for (int j = !((k % 2) ^ (i % 2)); j < (N / 2) - 1 + !((k % 2) ^ (i % 2)); j++) {
//                         B[i * N / 2 + j] = (A[(i - 1) * N / 2 + j] + A[i * N / 2 + j] + A[(i + 1) * N / 2 + j] + A[i * N / 2 + j + 1 - 2 * !((k % 2) ^ (i % 2))]) * 0.25f;
//                     }
//                 }
//             }
            
//         }

//         #pragma omp parallel for
//         for (int i = 0; i < N; i++) { 
//             for (int j = (i % 2); j < N - 1 + (i % 2); j += 2) {
//                 p[i * N + j] = A[i * N / 2 + j / 2];
//             }
//         }
//         #pragma omp parallel for
//         for (int i = 0; i < N; i++) {
//             for (int j = 1 - (i % 2); j < N - (i % 2); j += 2) {
//                 p[i * N + j] = B[i * N / 2 + j / 2];
//             }
//         }
        
//         free(A);
//         free(B);
//     }
// }

// void impl(int N, int step, double *p) {
//     if (N % 2 == 1) {
//         for (int k = 0; k < step; k++) {
//             omp_set_num_threads(14);
//             #pragma omp parallel for
//             for (int i = 1; i < N - 1; i++) {
//                 for (int j = 2 - ((k % 2) ^ (i % 2)); j < N - 1; j += 2) {
//                     p[i * N + j] = (p[(i - 1) * N + j] + p[(i + 1) * N + j] +
//                                         p[i * N + j + 1] + p[i * N + j - 1]) /
//                                         4.0f;
//                 }
//             }
//         }
//     } else {
//         double *A = (double *)aligned_alloc(32, (N/2) * N * sizeof(double));
//         double *B = (double *)aligned_alloc(32, (N/2) * N * sizeof(double));
        
//         omp_set_num_threads(14);
//         #pragma omp parallel for
//         for (int i = 0; i < N; i++) { 
//             for (int j = (i % 2); j < N - 1 + (i % 2); j += 2) {
//                 A[i * N / 2 + j / 2] = p[i * N + j];
//             }
//         }
//         #pragma omp parallel for
//         for (int i = 0; i < N; i++) {
//             for (int j = 1 - (i % 2); j < N - (i % 2); j += 2) {
//                 B[i * N / 2 + j / 2] = p[i * N + j];
//             }
//         }
//         // write_array_to_file("debug.txt", A, N, N/2, "A");
//         // write_array_to_file("debug.txt", B, N, N/2, "B");
//         // write_array_to_file("debug.txt", p, N, N, "P");

//         // int *index_mat = (int *)calloc(N * N, sizeof(int));
//         int k;
//         for (k = 0; k < step; k++) {
//             if (k % 2 == 0) {
//                 #pragma omp parallel for
//                 for (int i = 1; i < N - 1; i++) {
//                     int j;
//                     for (j = !((k % 2) ^ (i % 2)); j < (N / 2) - 1 + !((k % 2) ^ (i % 2)) - 3; j += 4) {
//                         __m256d t = _mm256_loadu_pd(&B[(i - 1) * N / 2 + j]);
//                         __m256d b = _mm256_loadu_pd(&B[(i + 1) * N / 2 + j]);
//                         __m256d l = _mm256_loadu_pd(&B[i * N / 2 + j]);
//                         __m256d r = _mm256_loadu_pd(&B[i * N / 2 + j + 1 - 2 * !((k % 2) ^ (i % 2))]);

//                         __m256d result = _mm256_add_pd(t, b);
//                         result = _mm256_add_pd(result, l);
//                         result = _mm256_add_pd(result, r);
//                         result = _mm256_mul_pd(result, _mm256_set1_pd(0.25));
//                         _mm256_storeu_pd(&A[i * N / 2 + j], result);
                        
//                     }
//                     for (; j < (N / 2) - 1 + !((k % 2) ^ (i % 2)); j++) {
//                         A[i * N / 2 + j] = (B[(i - 1) * N / 2 + j] + B[i * N / 2 + j] + B[(i + 1) * N / 2 + j] + B[i * N / 2 + j + 1 - 2 * !((k % 2) ^ (i % 2))]) * 0.25f;
//                     }
//                 }
//             } else {
//                 #pragma omp parallel for
//                 for (int i = 1; i < N - 1; i++) {
//                     int j;
//                     for (j = !((k % 2) ^ (i % 2)); j < (N / 2) - 1 + !((k % 2) ^ (i % 2)) - 3; j += 4) {
//                         __m256d t = _mm256_loadu_pd(&A[(i - 1) * N / 2 + j]);
//                         __m256d b = _mm256_loadu_pd(&A[(i + 1) * N / 2 + j]);
//                         __m256d l = _mm256_loadu_pd(&A[i * N / 2 + j]);
//                         __m256d r = _mm256_loadu_pd(&A[i * N / 2 + j + 1 - 2 * !((k % 2) ^ (i % 2))]);

//                         __m256d result = _mm256_add_pd(t, b);
//                         result = _mm256_add_pd(result, l);
//                         result = _mm256_add_pd(result, r);
//                         result = _mm256_mul_pd(result, _mm256_set1_pd(0.25));
//                         _mm256_storeu_pd(&B[i * N / 2 + j], result);
                        
//                     }
//                     for (; j < (N / 2) - 1 + !((k % 2) ^ (i % 2)); j++) {
//                         B[i * N / 2 + j] = (A[(i - 1) * N / 2 + j] + A[i * N / 2 + j] + A[(i + 1) * N / 2 + j] + A[i * N / 2 + j + 1 - 2 * !((k % 2) ^ (i % 2))]) * 0.25f;
//                     }
//                 }
//             }
//         }

//         #pragma omp parallel for
//         for (int i = 0; i < N; i++) { 
//             for (int j = (i % 2); j < N - 1 + (i % 2); j += 2) {
//                 p[i * N + j] = A[i * N / 2 + j / 2];
//             }
//         }
//         #pragma omp parallel for
//         for (int i = 0; i < N; i++) {
//             for (int j = 1 - (i % 2); j < N - (i % 2); j += 2) {
//                 p[i * N + j] = B[i * N / 2 + j / 2];
//             }
//         }
        
//         free(A);
//         free(B);
//     }
// }