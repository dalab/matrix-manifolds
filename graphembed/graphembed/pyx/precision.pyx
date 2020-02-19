# distutils: sources=graphembed/pyx/impl/precision.cpp
# distutils: language=c++
# cython: language_level=3
# cython: boundscheck=False, wraparound=False, cdivision=True
cimport cython

from libcpp.memory cimport make_unique, unique_ptr
from libcpp.unordered_set cimport unordered_set
from libcpp.vector cimport vector
from libcpp.string cimport string

import numpy as np


ctypedef fused float_dtype:
    double
    float


cdef extern from "impl/precision.hpp" namespace "graphembed":
    cdef cppclass GNodeDist nogil:
        int node, dist

    cdef cppclass StatsR nogil:
        vector[double] means, stds

    cdef cppclass FastPrecision nogil:
        FastPrecision(vector[unordered_set[int]]) except +
        FastPrecision(vector[vector[GNodeDist]]) except +

        double MeanAveragePrecision(const double*) const
        double MeanAveragePrecision(const float*) const
        StatsR LayerMeanF1Scores(const double*, size_t, size_t, size_t) const
        StatsR LayerMeanF1Scores(const float*, size_t, size_t, size_t) const
        StatsR LayerMeanAverageF1Scores(const double*, size_t) const
        StatsR LayerMeanAverageF1Scores(const float*, size_t) const

        # Function added later to avoid re-implementing the logic for summing up
        # all nodes on each layer of the shortest-path trees.
        vector[int] NodesPerLayer() const


cdef class PyFastPrecision:

    cdef int n, n_pdists
    # We have to allocate it on the heap because there's no way to
    # stack-construct it.
    cdef unique_ptr[FastPrecision] fast_precision

    def __cinit__(self, g):
        self.n = g.number_of_nodes()
        self.n_pdists = self.n * (self.n - 1) // 2
        self.fast_precision = self._init_fast_precision(g)

    cpdef double mean_average_precision(self, float_dtype[:] mpdists):
        with nogil:
            return self.fast_precision.get().MeanAveragePrecision(&mpdists[0])

    cpdef layer_mean_f1_scores(self, float_dtype[:] mpdists,
                               size_t num_pdists_sets=1,
                               size_t min_degree=1, size_t max_degree=99999):
        assert len(mpdists) == self.n_pdists * num_pdists_sets

        means = np.empty(self.n - 1)
        stds = np.empty(self.n - 1)
        cdef:
            double[:] means_view = means, stds_view = stds
            StatsR ret
            size_t i, n_ret

        with nogil:
            ret = self.fast_precision.get().LayerMeanF1Scores(
                    &mpdists[0], num_pdists_sets, min_degree, max_degree)
            n_ret = ret.means.size()
            for i in range(n_ret):
                means_view[i] = ret.means[i]
                stds_view[i] = ret.stds[i]

        return means[:n_ret], stds[:n_ret]

    cpdef layer_mean_average_f1_scores(self, float_dtype[:] mpdists,
                                       size_t num_pdists_sets=1):
        assert len(mpdists) == self.n_pdists * num_pdists_sets

        means = np.empty(self.n - 1)
        stds = np.empty(self.n - 1)
        cdef:
            double[:] means_view = means, stds_view = stds
            StatsR ret
            size_t i, n_ret

        with nogil:
            ret = self.fast_precision.get().LayerMeanAverageF1Scores(
                    &mpdists[0], num_pdists_sets)
            n_ret = ret.means.size()
            for i in range(n_ret):
                means_view[i] = ret.means[i]
                stds_view[i] = ret.stds[i]

        return means[:n_ret], stds[:n_ret]

    cpdef nodes_per_layer(self):
        npl_arr = np.empty(self.n - 1, dtype=np.int32)
        cdef:
            int[:] npl_view = npl_arr
            vector[int] npl_vec
            size_t i, n_ret

        with nogil:
            npl_vec = self.fast_precision.get().NodesPerLayer()
            n_ret = npl_vec.size()
            for i in range(n_ret):
                npl_view[i] = npl_vec[i]

        return npl_arr[:n_ret]

    cdef unique_ptr[FastPrecision] _init_fast_precision(self, g):
        if 'weight' in list(g.edges(data=True))[0][2]:
            return self._init_weighted(g)
        else:
            return self._init_unweighted(g)

    cdef unique_ptr[FastPrecision] _init_unweighted(self, g):
        cdef vector[unordered_set[int]] adj_list

        adj_list.resize(self.n)
        for u, v in g.edges():
            adj_list[u].insert(v)
            if not g.is_directed():
                adj_list[v].insert(u)

        with nogil:
            return make_unique[FastPrecision](adj_list)

    cdef unique_ptr[FastPrecision] _init_weighted(self, g):
        cdef vector[vector[GNodeDist]] adj_list
        cdef GNodeDist node_dist

        adj_list.resize(self.n)
        for u, v, w in g.edges(data='weight'):
            node_dist.node = v
            node_dist.dist = w
            adj_list[u].push_back(node_dist)
            if not g.is_directed():
                node_dist.node = u
                adj_list[v].push_back(node_dist)

        with nogil:
            return make_unique[FastPrecision](adj_list)
