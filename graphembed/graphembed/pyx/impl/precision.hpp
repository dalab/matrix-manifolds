#ifndef GRAPHEMBED_PYX_IMPL_PRECISION_HPP
#define GRAPHEMBED_PYX_IMPL_PRECISION_HPP

#include <memory>
#include <unordered_set>
#include <vector>

namespace graphembed {

template <typename T>
struct NodeDist {
  int node;
  T dist;
};
using AdjList = std::vector<std::unordered_set<int>>;
using GNodeDist = NodeDist<int>;  // We assume integer graph weights!
using WeightedAdjList = std::vector<std::vector<GNodeDist>>;

// Convenient return type that aggregates the means and standard deviations of a
// compute metric.
struct StatsR {
  std::vector<double> means;
  std::vector<double> stds;
};

class FastPrecision {
 public:
  FastPrecision(AdjList adj_list);
  FastPrecision(WeightedAdjList adj_list);
  ~FastPrecision();  // Default dtor, but needs to be defined due to PIMPL.

  // Returns the number of nodes on each layer, accumulated across all
  // shortest-path trees.
  std::vector<int> NodesPerLayer() const;

  // Returns the mean of the per-node average precisions.
  // NOTE: This is defined for unweighted graphs only.
  double MeanAveragePrecision(const double* mpdists) const;
  double MeanAveragePrecision(const float* mpdists) const;

  // Compute the mean F1 scores of all nodes on a shortest-path tree layer.
  StatsR LayerMeanF1Scores(
      const double* mpdists, size_t num_pdists_sets, size_t min_degree = 1,
      size_t max_degree = std::numeric_limits<size_t>::max()) const;
  StatsR LayerMeanF1Scores(
      const float* mpdists, size_t num_pdists_sets, size_t min_degree = 1,
      size_t max_degree = std::numeric_limits<size_t>::max()) const;

  // Compute the mean average F1 scores. The `mean' is taken across the nodes in
  // the tree and the `average' is taken across each nodes's neighbors that are
  // `k' hops away.
  StatsR LayerMeanAverageF1Scores(const double* mpdists,
                                  size_t num_pdists_sets) const;
  StatsR LayerMeanAverageF1Scores(const float* mpdists,
                                  size_t num_pdists_sets) const;

 private:
  class FastPrecisionImpl;
  std::unique_ptr<const FastPrecisionImpl> pimpl_;
};

}  // namespace graphembed

#endif  // GRAPHEMBED_PYX_IMPL_PRECISION_HPP
