# cython: language_level=3
# distutils: language=c++

from libcpp cimport bool
from libcpp.vector cimport vector

from lpsmap.ad3qp.base cimport PGenericFactor


cdef class PFactorSequence(PGenericFactor):

    def __cinit__(self, bool allocate=True):
        self.allocate = allocate
        if allocate:
           self.thisptr = new FactorSequence()

    def __dealloc__(self):
        if self.allocate:
            del self.thisptr

    def initialize(self, vector[int] num_states):
        (<FactorSequence*>self.thisptr).Initialize(num_states)


cdef class PFactorSequenceBinary(PGenericFactor):

    def __cinit__(self, bool allocate=True):
        self.allocate = allocate
        if allocate:
           self.thisptr = new FactorSequenceBinary()

    def __dealloc__(self):
        if self.allocate:
            del self.thisptr

    def initialize(self, int length):
        (<FactorSequenceBinary*>self.thisptr).Initialize(length)
