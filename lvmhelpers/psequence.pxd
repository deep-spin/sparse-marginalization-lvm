# cython: language_level=3
# distutils: language=c++

from libcpp cimport bool
from libcpp.vector cimport vector

from lpsmap.ad3qp.base cimport Factor, GenericFactor, PGenericFactor

cdef extern from "sequence.h" namespace "AD3":

    cdef cppclass FactorSequence(GenericFactor):
        FactorSequence()
        void Initialize(vector[int] num_states)


cdef extern from "sequence_binary.h" namespace "AD3":

    cdef cppclass FactorSequenceBinary(FactorSequence):
        FactorSequenceBinary()
        void Initialize(int length)
