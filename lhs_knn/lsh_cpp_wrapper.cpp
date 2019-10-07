// Code based off of https://github.com/ariskag/cuLSH

// to compile for .so file run the following: 
// nvcc -O4 -D_FORCE_INLINES -L/usr/local/cuda/lib64 --shared -o lsh_lib.so -Xcompiler -fPIC --x cu cpp_wrapper.cpp -lcuda -lcublas -lcudart -lcurand -Wno-deprecated-gpu-targets

#include "cuLSH.h"

/*
void* hashtables    pointer to reused hashtables
int D;              # of dimensions
int N;              # of data points
int L;              # of tables
int M;              # of projection dimensions
float W;            bucket width
float* matrix;      dataset ([D x N] stored columnwise)
*/
extern "C" void * create_hash(void* hashtables, int N, int D, int L, int M, float W, float* matrix)
{
    // Initialize hashtable if not exist
    if(hashtables == NULL){
        hashtables = new cuLSH::HashTables;
    }
    
    // Reset hashtable
    if(!(*(cuLSH::HashTables*)hashtables).reset(N, D, L, M, W, NULL)) {
        printf("\nGPU LHS Table failed to reset...\n");
    }
    
    // Populate hashtable
    if(!(*(cuLSH::HashTables*)hashtables).index(matrix, NULL)) {
        printf("\nGPU LHS Table failed to populate...\n");
    }

    return (void *) hashtables;
}

/*
void* hashtables    pointer to populated (indexed) hashtables
int K;              # of neighbors
int T;              # of probing bins
int Q;              # of query points
float* matrix;      dataset ([D x N] stored columnwise)
float* queries;     queries ([D x Q] stored columnwise)
*/
extern "C" void perform_search(void* hashtables, int K, int T, int Q, float* matrix, float* queries, int * res)
{
    // Initialize search table
    cuLSH::SearchTables searchtables;

    // Reset searchtable
    searchtables.reset((cuLSH::HashTables*) hashtables, K, T);

    // Perform LSH KNN
    if(!searchtables.search(queries, Q, matrix, NULL)) {
        printf("\nGPU LSH find failed...\n");
    }
    
    // Extract neighbors
    const int * t_res = searchtables.getKnnIds();
    const float* distances = searchtables.getKnnDistances();
    
    std::memcpy(res, t_res, Q * sizeof(int));
}








