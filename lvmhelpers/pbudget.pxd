# cython: language_level=3
# distutils: language=c++

from libcpp cimport bool
from libcpp.vector cimport vector

from lpsmap.ad3qp.base cimport Factor, GenericFactor, PGenericFactor

from .pbernoulli cimport FactorBernoulli

cdef extern from "budget.h" namespace "AD3":

    cdef cppclass FactorBudget(FactorBernoulli):
        FactorBudget()
        void Initialize(int length, int budget)
