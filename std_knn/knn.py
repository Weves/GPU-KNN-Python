import ctypes

testlib = ctypes.CDLL('/datasets/home/20/220/ckweaver/CSE198/kNN-CUDA/code/knnlib.so')

def gpu_knn(ref, query, dim, k):

    ref_arr = (ctypes.c_float * len(ref))(*ref)
    query_arr = (ctypes.c_float * len(query))(*query)
    ref_len = ctypes.c_int(int(len(ref) / dim))
    query_len = ctypes.c_int(int(len(query) / dim))
    c_dim = ctypes.c_int(dim)
    k = ctypes.c_int(k)
    res = (ctypes.c_int * int(len(query) / dim))()

    testlib.knn_gpu.argtypes = [ctypes.POINTER(ctypes.c_float), 
                                ctypes.c_int, ctypes.POINTER(ctypes.c_float),
                                ctypes.c_int, ctypes.c_int, ctypes.c_int, 
                                ctypes.POINTER(ctypes.c_int)]

    testlib.knn_gpu(ref_arr, ref_len, query_arr, query_len, c_dim, k, res)
    
    return res

if __name__ == '__main__':
   
    
    ref = [1.0,5.0,4.0,1.0,5.0,3.0]
    query = [5.0,1.0,4.0,5.0,1.0,4.0]
    
    res = gpu_knn(ref, query, 2, 1)
    print(res[0])
    print(res[1])
    print(res[2])



