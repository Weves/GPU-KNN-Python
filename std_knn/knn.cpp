#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

#include "knncuda.h"

/*
In order t build shared libarary file, use the following 
nvcc --shared -o knnlib.so -Xcompiler -fPIC knn.cpp knncuda.cu -lcuda -lcublas -lcudart -Wno-deprecated-gpu-targets
*/

/*
 * @param ref            reference points
 * @param ref_nb         number of reference points
 * @param query          query points
 * @param query_nb       number of query points
 * @param dim            dimension of reference and query points
 * @param k              number of neighbors to consider
 * @param res            pointer to results
 */

extern "C" void * knn_gpu(const float * ref,
             int           ref_nb,
             const float * query,
             int           query_nb,
             int           dim,
             int           k,
             int * res) {
    
    // Allocate memory for computed distances
    float * test_knn_dist  = (float*) malloc(query_nb * k * sizeof(float));
    
    knn_cuda_global(ref, ref_nb, query, query_nb, dim, k, test_knn_dist, res);
    
    free(test_knn_dist);
    
}

