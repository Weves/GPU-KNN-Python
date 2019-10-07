import ctypes

cuda_lsh_lib = ctypes.CDLL('./lsh_lib.so')

# hash_ptr        void pointer for hashtables
# data_dim        dimension of data
# num_tables      number of tables to average hash over
# proj_dim        number of projection dimensions
# bucket_wid      bucket width, float
def lsh_hash(hash_ptr, data_dim, num_tables, proj_dim, bucket_wid, refs):
    
    hashtables = ctypes.c_void_p(hash_ptr)
    N = ctypes.c_int(int(len(refs)/data_dim))
    D = ctypes.c_int(data_dim)
    L = ctypes.c_int(num_tables)
    M = ctypes.c_int(proj_dim)
    W = ctypes.c_float(bucket_wid)
    matrix = (ctypes.c_float * len(refs))(*refs)
    
    cuda_lsh_lib.create_hash.argtypes = [ctypes.c_void_p,                   \
                                         ctypes.c_int,                      \
                                         ctypes.c_int,                      \
                                         ctypes.c_int,                      \
                                         ctypes.c_int,                      \
                                         ctypes.c_float,                    \
                                         ctypes.POINTER(ctypes.c_float)     \
                                        ]
    
    cuda_lsh_lib.create_hash.restype = ctypes.c_ulonglong
    
    return cuda_lsh_lib.create_hash(hashtables, N, D, L, M, W, matrix)


# hash_ptr        void pointer for hashtables
# num_neigh       number of nearest neighbors to return
# probe_bin       number of probing bins
# num_query       number of query points
# refs            reference points matrix
# queries         query points matrix
def lsh_search(hash_ptr, num_neigh, probe_bin, num_query, refs, queries):
    
    hashtables = ctypes.c_ulonglong(hash_ptr)
    K = ctypes.c_int(num_neigh)
    T = ctypes.c_int(probe_bin)
    Q = ctypes.c_int(num_query)
    matrix = (ctypes.c_float * len(refs))(*refs)
    queries = (ctypes.c_float * len(queries))(*queries)
    res = (ctypes.c_int * int(num_query))()
    
    cuda_lsh_lib.perform_search.argtypes = [ctypes.c_ulonglong,         \
                                     ctypes.c_int,                      \
                                     ctypes.c_int,                      \
                                     ctypes.c_int,                      \
                                     ctypes.POINTER(ctypes.c_float),    \
                                     ctypes.POINTER(ctypes.c_float),    \
                                     ctypes.POINTER(ctypes.c_int),      \
                                    ]
    
    cuda_lsh_lib.perform_search.restype = (ctypes.c_int * int(num_query))
    
    cuda_lsh_lib.perform_search(hashtables, K, T, Q, matrix, queries, res)
    
    return res


if __name__ == '__main__':
    
    # (1, 5), (4, 1), (5, 3)
    ref =   [1.0, 5.0, 4.0, 1.0, 5.0, 3.0]
    # (5, 1), (4, 5), (1, 4)
    query = [5.0, 1.0, 4.0, 5.0, 1.0, 4.0]
    
    data_dim = 2
    q_len = int(len(query) / data_dim)
    k_neigh = 1
    hash_ptr = 0
    
    num_tables = 100
    projection_dim = 10
    bucket_wid = 5
    probe_bins = 3
    
    hash_tables = c_pointer()
    
    a = lsh_hash(hash_ptr, data_dim, num_tables, projection_dim, bucket_wid, ref)
    results = lsh_search(a, k_neigh, probe_bins, q_len, ref, query)
    
    # correct = 1 2 0
    print(results[0])
    print(results[1])
    print(results[2])




