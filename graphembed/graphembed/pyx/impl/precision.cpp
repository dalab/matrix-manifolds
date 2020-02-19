#include "graphembed/pyx/impl/precision.hpp"

#include <algorithm>
#include <cassert>
#include <execution>
#include <queue>
#include <valarray>
#include <variant>
#include <vector>

#include <boost/container/flat_set.hpp>  // Used for ordered statistics.

namespace graphembed {
namespace {

// A convenient parallel map on the vector [0,1,...,n-1].
template <typename F>
auto parallel_map(int n, F op) {
  std::vector<std::invoke_result_t<F, int>> out(n);
  std::vector<int> indices(n);
  std::iota(indices.begin(), indices.end(), 0);
  std::transform(std::execution::par, indices.cbegin(), indices.cend(),
                 out.begin(), op);
  return out;
}

// A convenient parallel map-reduce on the vector [0,1,...,n-1].
template <typename UnaryOp, typename T = std::invoke_result_t<UnaryOp, int>,
          typename BinaryOp = std::plus<T>>
auto parallel_map_reduce(int n, UnaryOp unary, T init, BinaryOp binary = {}) {
  std::vector<int> indices(n);
  std::iota(indices.begin(), indices.end(), 0);
  return std::transform_reduce(std::execution::par, indices.cbegin(),
                               indices.cend(), std::forward<T>(init), binary,
                               unary);
}

// Computes shortest paths on a graph given via an adjacency list. It has two
// specializations (see below).
template <typename T>
std::vector<int> ShortestPaths(int u, const T& adj_list);

// Shortest paths in unweighted graph via BFS.
template <>
std::vector<int> ShortestPaths<AdjList>(int u, const AdjList& adj_list) {
  std::vector<int> dists(adj_list.size(), -1);
  std::queue<int> bfs_q;

  bfs_q.emplace(u);
  dists[u] = 0;
  while (!bfs_q.empty()) {
    int node = bfs_q.front();
    bfs_q.pop();

    for (auto neigh : adj_list[node]) {
      if (dists[neigh] == -1) {
        bfs_q.emplace(neigh);
        dists[neigh] = dists[node] + 1;
      }
    }
  }
  return dists;
}

// Shortest paths in weighted graph via Dijkstra's Algorithm.
template <>
std::vector<int> ShortestPaths<WeightedAdjList>(
    int u, const WeightedAdjList& adj_list) {
  std::vector<int> dists(adj_list.size(), std::numeric_limits<int>::max());
  const auto cmp = [](const auto& lhs, const auto& rhs) {
    return lhs.dist > rhs.dist;  // min-heap
  };
  std::priority_queue<GNodeDist, std::vector<GNodeDist>, decltype(cmp)> q(cmp);

  dists[u] = 0;
  q.push({u, 0});
  while (!q.empty()) {
    auto [node, dist] = q.top();
    q.pop();

    if (dist == dists[node]) {
      for (auto [neigh, cost] : adj_list[node]) {
        auto new_dist = dist + cost;
        if (new_dist < dists[neigh]) {
          dists[neigh] = new_dist;
          q.push({neigh, new_dist});
        }
      }
    }
  }
  return dists;
}

// Tags each distance with its corresponding node (assumed to be in order, from
// 0 to n-1) and sorts them.
template <typename T>  // T is LessThanComparable; no concepts in LLVM :-(
std::vector<NodeDist<T>> SortNodeDists(const std::vector<T>& dists) {
  int n = dists.size();
  std::vector<NodeDist<T>> sorted_dists(n);

  for (int i = 0; i < n; ++i) {
    sorted_dists[i].node = i;
    sorted_dists[i].dist = dists[i];
  }
  std::sort(
      std::execution::par, sorted_dists.begin(), sorted_dists.end(),
      [&](const auto& lhs, const auto& rhs) { return lhs.dist < rhs.dist; });

  return sorted_dists;
}

}  // namespace

class FastPrecision::FastPrecisionImpl {
 public:
  // AdjList for unweighted graphs and WeightedAdjList for graphs with costs.
  // The only member passed on construction.
  const std::variant<AdjList, WeightedAdjList> adj_list_;

  // Number of nodes in the graph, for convenience.
  //
  // USES: adj_list_
  const int n_nodes_ =
      std::visit([](auto&& adj_list) { return adj_list.size(); }, adj_list_);

  // The node degrees.
  //
  // USES: n_nodes_
  const std::vector<size_t> node_degrees_ =
      parallel_map(n_nodes_, [this](int u) {
        return std::visit([u](auto&& adj_list) { return adj_list[u].size(); },
                          adj_list_);
      });

  // Number of pairwise distances in the graph, for convenience.
  //
  // USES: n_nodes_
  const int n_pdists_ = (n_nodes_ * (n_nodes_ - 1)) / 2;

  // Lists of nodes sorted by distances to the source.
  //
  // USES: n_nodes_, adj_list_
  const std::vector<std::vector<GNodeDist>> sorted_dists_ = std::visit(
      [this](auto&& adj_list) {
        return parallel_map(n_nodes_, [&adj_list](int u) {
          return SortNodeDists(ShortestPaths(u, adj_list));
        });
      },
      adj_list_);

  // For each node ``u``, the position ``v`` stores the layer of node ``v`` in
  // the shortest path tree rooted at ``u``, starting from 0 at the root (i.e.,
  // node ``u`` itself).
  //
  // USES: n_nodes_, sorted_dists_
  const std::vector<std::vector<int>> node_layers_ =
      parallel_map(n_nodes_, [this](int u) {
        std::vector<int> layers(n_nodes_, 0);
        int layer = 0;
        for (int i = 1; i < n_nodes_; ++i) {  // Layer of node ``u`` is 0.
          if (sorted_dists_[u][i].dist > sorted_dists_[u][i - 1].dist) {
            layer += 1;
          }
          layers[sorted_dists_[u][i].node] = layer;
        }
        return layers;
      });

  // For each node ``u``, the position ``l`` stores the beginning and the end of
  // the interval of that layer in sorted-nodes indices.
  //
  // USES: n_nodes_, sorted_dists_
  const std::vector<std::vector<std::pair<int, int>>> layer_intervals_ =
      parallel_map(n_nodes_, [this](int u) {
        std::vector<std::pair<int, int>> intervals;
        intervals.emplace_back(0, -1);
        for (int i = 1; i < n_nodes_; ++i) {
          if (sorted_dists_[u][i].dist > sorted_dists_[u][i - 1].dist) {
            intervals.back().second = i;    // close the previous one
            intervals.emplace_back(i, -1);  // open the new one
          }
        }
        intervals.back().second = n_nodes_;  // close the last one
        return intervals;
      });

  // The diameter of the graph defined as the maximum number of layers across
  // all shortest-path trees. Note that *for weighted graphs*, this might be
  // different than the usual definition of the diameter.
  //
  // USES: layer_intervals_
  const size_t max_num_layers_ =
      std::max_element(layer_intervals_.begin(), layer_intervals_.end(),
                       [](const auto& lhs, const auto& rhs) {
                         return lhs.size() < rhs.size();
                       }) -> size();

  // For each layer, position ``l`` stores the number of nodes on that layer
  // summed across all shortest-path trees.
  //
  // NOTE: We use `std::valarray`s wherever we need component-wise operations.
  // For instance, here we need to add the number of nodes per layer for each
  // shortest-path tree.
  //
  // USES: max_num_layers_, layer_intervals_
  const std::valarray<int> nodes_per_layer_ = parallel_map_reduce(
      n_nodes_,
      [&](int u) {
        std::valarray<int> nodes_per_layer(0, max_num_layers_);
        for (auto i = 0u; i < layer_intervals_[u].size(); ++i) {
          auto [s, e] = layer_intervals_[u][i];
          nodes_per_layer[i] = e - s;
        }
        return nodes_per_layer;
      },
      /*init=*/std::valarray<int>(0, max_num_layers_));

  FastPrecisionImpl(std::variant<AdjList, WeightedAdjList> adj_list)
      : adj_list_(std::move(adj_list)) {}

  // Returns a vector of ``n_nodes_`` entries representing the distances from
  // node ``u`` to all other nodes (including itself: the distance should be 0).
  //
  // USES: n_nodes_
  template <typename T>
  std::vector<double> ExtractDistancesFromCondensedPdists(
      int u, const T* mpdists) const {
    const auto condensed_index = [&](int u, int v) {
      if (u > v) std::swap(u, v);
      return n_nodes_ * u - u * (u + 1) / 2 + (v - u) - 1;
    };

    std::vector<double> dists(n_nodes_);
    for (int i = 0; i < n_nodes_; ++i) {
      if (i == u) {
        dists[i] = 0;
      } else {
        dists[i] = mpdists[condensed_index(u, i)];
        assert(dists[i] > 0);
      }
    }
    return dists;
  }

  /////////////////////////////////////////////////////////////////////////////
  // Mean Average Precision

  // USES: n_nodes_, adj_list_
  template <typename T>
  double AveragePrecision(int u, const T* mpdists) const {
    // We limit this to unweighted graphs. Let it raise here if using it on
    // weighted graphs.
    const auto& adj_list = std::get<AdjList>(adj_list_);

    // Extract the sorted distances to all other nodes.
    const auto manifold_sorted_neighs =
        SortNodeDists(ExtractDistancesFromCondensedPdists(u, mpdists));

    const auto& neighs = adj_list[u];
    int n_neighs = neighs.size(), n_correct = 0;
    double precisions_sum = 0;

    // Start from 1 because ``u`` itself should be on 0.
    assert(manifold_sorted_neighs[0].node == u);
    for (int i = 1; i < n_nodes_; ++i) {
      if (neighs.find(manifold_sorted_neighs[i].node) != neighs.end()) {
        n_correct += 1;
        precisions_sum += static_cast<double>(n_correct) / i;
        if (n_correct == n_neighs) {
          break;
        }
      }
    }
    return precisions_sum / n_neighs;
  }

  // USES: AveragePrecision, n_nodes_
  template <typename T>
  std::vector<double> AveragePrecisions(const T* mpdists) const {
    return parallel_map(n_nodes_,
                        [&](int u) { return AveragePrecision(u, mpdists); });
  }

  // USES: AveragePrecisions, n_nodes_
  template <typename T>
  double MeanAveragePrecision(const T* mpdists) const {
    auto aps = AveragePrecisions(mpdists);
    // The overhead of parallelization might be more than the savings for a
    // vector of thousands of entries, so we do it sequentially.
    return std::reduce(std::execution::seq, aps.begin(), aps.end()) / n_nodes_;
  }

  /////////////////////////////////////////////////////////////////////////////
  // Layer-mean F1 Scores

  struct MomentsWithCounts {
    std::valarray<double> m1, m2;
    std::valarray<int> counts;

    MomentsWithCounts(int n) : m1(0.0, n), m2(0.0, n), counts(0, n) {}

    MomentsWithCounts operator+(MomentsWithCounts other) const {
      other.m1 += m1;
      other.m2 += m2;
      other.counts += counts;
      return other;
    }
  };

  // NOTE: For "aristocratic nodes", where most other nodes are on the first
  // layer, their order is irrelevant and most of them will be assigned an F1
  // score of roughly 1. Because of that, we see large standard deviations in
  // the F1@1 metric, and the mean is pulled to higher values. In the thesis, I
  // did not have time to re-aggregate all the results and I considered that
  // this impacts embedding spaces in the same way. But for a more careful
  // analysis, I would have to do something about this "issue". One option I
  // thought about is to not include the nodes with a degree over some threshold
  // in the average.

  // USES: n_nodes_, max_num_layers_, node_layers_, layer_intervals_
  template <typename T>
  MomentsWithCounts LayerF1Scores(int u, const T* mpdists) const {
    const auto manifold_sorted_neighs =
        SortNodeDists(ExtractDistancesFromCondensedPdists(u, mpdists));

    // All aggregators are initialized to 0.
    MomentsWithCounts ret(max_num_layers_ - 1);

    // As we iterate through the nodes in manifold-sorted order (`i = 1,...,n`)
    // we want to know how many of the previously seen nodes (`k = 1,...,i-1`)
    // appear "before" the current node in the reference ordering given by the
    // shortest-path tree rooted at ``u``.
    const auto ord = [this, u](int v1, int v2) {
      return node_layers_[u][v1] < node_layers_[u][v2];
    };
    boost::container::flat_multiset<int, decltype(ord)> ordered_nodes(ord);
    ordered_nodes.reserve(n_nodes_ - 1);

    assert(manifold_sorted_neighs[0].node == u);  // skip 0 which is ``u``
    for (int i = 1; i < n_nodes_; ++i) {
      int v = manifold_sorted_neighs[i].node;
      size_t layer = node_layers_[u][v];
      auto insert_pos = ordered_nodes.insert(v);
      // +1 to include self (consistent with recall below).
      auto nodes_before = insert_pos - ordered_nodes.cbegin() + 1;  // >= 1
      assert(nodes_before <= i);
      // The precision is the number of nodes that appear before ``v`` on *both*
      // the manifold and the input-graph orderings divided by the former.
      auto precision = static_cast<double>(nodes_before) / i;

      auto [actual_nodes_strict_before, _] = layer_intervals_[u][layer];
      // Ignore the root of the shortest-path tree.
      actual_nodes_strict_before -= 1;
      // Adjust for the nodes on the same layer seen so far.
      auto lb = ordered_nodes.lower_bound(v);
      auto nodes_same_layer_so_far = insert_pos - lb;
      // +1 to include self: avoid division by zero.
      auto actual_nodes_before =
          actual_nodes_strict_before + nodes_same_layer_so_far + 1;
      assert(nodes_before <= actual_nodes_before);
      // The recall is the number of nodes that appear before ``v`` on *both*
      // the manifold and the input-graph orderings divided by the latter.
      auto recall = static_cast<double>(nodes_before) / actual_nodes_before;

      auto f1 = 2 * precision * recall / (precision + recall);
      ret.m1[layer - 1] += f1;
      ret.m2[layer - 1] += f1 * f1;
      ret.counts[layer - 1] += 1;
    }
    return ret;
  }

  // USES: LayerF1Scores, n_nodes_, n_pdists_, max_num_layers_
  template <typename T>
  StatsR LayerMeanF1Scores(const T* mpdists, size_t num_pdists_sets,
                           size_t min_degree, size_t max_degree) const {
    const auto zero = MomentsWithCounts(max_num_layers_ - 1);
    // Total layer F1s (over all nodes and all repetitions).
    auto [f1_sums, f1_sq_sums, counts] = parallel_map_reduce(
        num_pdists_sets * n_nodes_,
        [&](int i) {
          int u = i % n_nodes_;
          if (node_degrees_[u] < min_degree || node_degrees_[u] > max_degree) {
            return zero;
          }
          int offset = (i / n_nodes_) * n_pdists_;
          return LayerF1Scores(u, mpdists + offset);
        },
        zero);

    // Average the scores and return them.
    std::vector<double> means(max_num_layers_ - 1), stds(max_num_layers_ - 1);
    for (auto i = 0u; i < max_num_layers_ - 1; ++i) {
      means[i] = f1_sums[i] / counts[i];
      stds[i] = f1_sq_sums[i] / counts[i] - means[i] * means[i];
    }
    return {means, stds};
  }

  template <typename T>
  StatsR LayerMeanAverageF1Scores(const T* mpdists,
                                  size_t num_pdists_sets) const {
    const auto zero = MomentsWithCounts(max_num_layers_ - 1);
    // Total layer average F1s.
    auto [f1_sums, f1_sq_sums, counts] = parallel_map_reduce(
        num_pdists_sets * n_nodes_,
        [&](int i) {
          int u = i % n_nodes_;
          int offset = (i / n_nodes_) * n_pdists_;
          auto mwc = LayerF1Scores(u, mpdists + offset);
          for (auto i = 0u; i < max_num_layers_ - 1; ++i) {
            if (mwc.counts[i] > 0) {
              mwc.m1[i] /= mwc.counts[i];
              mwc.counts[i] = 1;
            }
          }
          mwc.m2 = mwc.m1 * mwc.m1;
          return mwc;
        },
        zero);

    // Average the scores and return them.
    std::vector<double> means(max_num_layers_ - 1), stds(max_num_layers_ - 1);
    for (auto i = 0u; i < max_num_layers_ - 1; ++i) {
      means[i] = f1_sums[i] / counts[i];
      stds[i] = f1_sq_sums[i] / counts[i] - means[i] * means[i];
    }
    return {means, stds};
  }
};

FastPrecision::FastPrecision(AdjList adj_list)
    : pimpl_(std::make_unique<FastPrecisionImpl>(std::move(adj_list))) {}

FastPrecision::FastPrecision(WeightedAdjList adj_list)
    : pimpl_(std::make_unique<FastPrecisionImpl>(std::move(adj_list))) {}

FastPrecision::~FastPrecision() = default;

std::vector<int> FastPrecision::NodesPerLayer() const {
  return {std::cbegin(pimpl_->nodes_per_layer_),
          std::cend(pimpl_->nodes_per_layer_)};
}

double FastPrecision::MeanAveragePrecision(const double* mpdists) const {
  return pimpl_->MeanAveragePrecision(mpdists);
}

double FastPrecision::MeanAveragePrecision(const float* mpdists) const {
  return pimpl_->MeanAveragePrecision(mpdists);
}

StatsR FastPrecision::LayerMeanF1Scores(const double* mpdists,
                                        size_t num_pdists_sets,
                                        size_t min_degree,
                                        size_t max_degree) const {
  return pimpl_->LayerMeanF1Scores(mpdists, num_pdists_sets, min_degree,
                                   max_degree);
}

StatsR FastPrecision::LayerMeanF1Scores(const float* mpdists,
                                        size_t num_pdists_sets,
                                        size_t min_degree,
                                        size_t max_degree) const {
  return pimpl_->LayerMeanF1Scores(mpdists, num_pdists_sets, min_degree,
                                   max_degree);
}

StatsR FastPrecision::LayerMeanAverageF1Scores(const double* mpdists,
                                               size_t num_pdists_sets) const {
  return pimpl_->LayerMeanAverageF1Scores(mpdists, num_pdists_sets);
}
StatsR FastPrecision::LayerMeanAverageF1Scores(const float* mpdists,
                                               size_t num_pdists_sets) const {
  return pimpl_->LayerMeanAverageF1Scores(mpdists, num_pdists_sets);
}

}  // namespace graphembed
