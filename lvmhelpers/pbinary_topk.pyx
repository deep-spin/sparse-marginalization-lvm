# cython: language_level=3
# distutils: language=c++

from libcpp.vector cimport vector


cdef extern from "binary_topk.h":

    cdef cppclass scored_cfg:
        float score
        vector[int] cfg_vector(int dim)
        void populate_out(float* out, int dim)

    vector[scored_cfg] topk(vector[float] x, int k)
    vector[scored_cfg] topk(float* x, int size, int k)


cpdef binary_topk(vector[float] scores, int k):
    cdef vector[scored_cfg] out = topk(scores, k)
    return [(c.score, c.cfg_vector(scores.size())) for c in out]


def batched_topk(float[:, :] scores,
                 float[:, :, :] configs,
                 int k):

    cdef vector[scored_cfg] out
    cdef int size = scores.shape[1]
    cdef int i, j

    for i in range(scores.shape[0]):
        out = topk(&scores[i, 0], size, k)
        for j in range(k):
            out[j].populate_out(&configs[i, j, 0], size)
