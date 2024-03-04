#ifndef TREE_HPP
#define TREE_HPP

#include <vector>

namespace dmk {

struct TreeData {
    std::vector<bool> leaf_flag;
    std::vector<bool> in_flag;
    std::vector<bool> out_flag;
    std::vector<std::vector<int>> level_indices;
    std::vector<int> src_counts_local;
    std::vector<int> src_counts_global;

    template <typename TREE>
    TreeData(const TREE &tree, int ns);
};

} // namespace dmk

#endif
