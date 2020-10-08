# cython: language_level=3
# distutils: language=c++

from libcpp cimport bool
from libcpp.vector cimport vector

from lpsmap.ad3qp.base cimport Factor, GenericFactor, PGenericFactor


cdef class PFactorBernoulli(PGenericFactor):

    def __cinit__(self, bool allocate=True):
        self.allocate = allocate
        if allocate:
           self.thisptr = new FactorBernoulli()

    def __dealloc__(self):
        if self.allocate:
            del self.thisptr

    def initialize(self, int length):
        (<FactorBernoulli*>self.thisptr).Initialize(length)
