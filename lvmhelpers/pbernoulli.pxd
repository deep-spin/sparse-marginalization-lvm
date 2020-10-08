# cython: language_level=3
# distutils: language=c++

from libcpp cimport bool
from libcpp.vector cimport vector

from lpsmap.ad3qp.base cimport Factor, GenericFactor, PGenericFactor

cdef extern from "bernoulli.h" namespace "AD3":

    cdef cppclass FactorBernoulli(GenericFactor):
        FactorBernoulli()
        void Initialize(int length)
