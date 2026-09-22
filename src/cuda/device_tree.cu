// The one translation unit nvcc compiles; defines the interface declared in dmk/cuda/device_tree.hpp.

#include "sctl/tree.hpp"
#include "sctl/tree.txx"
#include "sctl/experimental/gpu-tree.hpp"
#include "dmk/cuda/device_tree.hpp"
#include "dmk/cuda/device_vector.hpp"

#include <thrust/device_ptr.h>

namespace dmk::cuda::device_tree {

namespace detail_deviceTree {

template <class T>
using Scratch = gpu_tree::DeviceScratch<T, DeviceVector>;
template <class T>
using View = gpu_tree::DataView<T, DeviceVector>;

template <class T>
const T *raw(const DeviceVector<T> &v) {
    return thrust::raw_pointer_cast(v.data());
}

} // namespace detail_deviceTree

template <class Real, Integer DIM>
struct PtTree<Real, DIM>::Impl {
    using Tree = gpu_tree::PtTree<Real, DIM, DeviceVector>;

    explicit Impl(const Comm &comm) : tree(comm) {}

    Tree tree;
};

template <class Real, Integer DIM>
PtTree<Real, DIM>::PtTree(const Comm &comm) : p_(new Impl(comm)) {}

template <class Real, Integer DIM>
PtTree<Real, DIM>::~PtTree() {
    delete p_;
}

template <class Real, Integer DIM>
void PtTree<Real, DIM>::UpdateRefinement(const Real *coord, Long n, MemSpace space, Long M, bool balance21,
                                         Periodicity periodicity, Integer halo_size) {
    if (space == MemSpace::Device) {
        p_->tree.UpdateRefinement(detail_deviceTree::View<const Real>{coord, n * DIM}, M, balance21, periodicity, halo_size);
        return;
    }
    detail_deviceTree::Scratch<Real> stage(n * DIM);
    gpu_tree::detail::hostToDevice(coord, n * DIM, stage.data());
    p_->tree.UpdateRefinement(stage, M, balance21, periodicity, halo_size);
}

template <class Real, Integer DIM>
void PtTree<Real, DIM>::AddParticles(const std::string &name, const Real *coord, Long n, MemSpace space) {
    if (space == MemSpace::Device) {
        p_->tree.AddParticles(name, detail_deviceTree::View<const Real>{coord, n * DIM});
        return;
    }
    detail_deviceTree::Scratch<Real> stage(n * DIM);
    gpu_tree::detail::hostToDevice(coord, n * DIM, stage.data());
    p_->tree.AddParticles(name, stage);
}

template <class Real, Integer DIM>
void PtTree<Real, DIM>::AddParticleData(const std::string &data_name, const std::string &particle_name,
                                        const Real *data, Long n, MemSpace space) {
    if (space == MemSpace::Device) {
        p_->tree.AddParticleData(data_name, particle_name, detail_deviceTree::View<const Real>{data, n});
        return;
    }
    detail_deviceTree::Scratch<Real> stage(n);
    gpu_tree::detail::hostToDevice(data, n, stage.data());
    p_->tree.AddParticleData(data_name, particle_name, stage);
}

template <class Real, Integer DIM>
void PtTree<Real, DIM>::AddParticleData(const std::string &data_name, const std::string &particle_name, Long dof) {
    p_->tree.AddParticleData(data_name, particle_name, dof);
}

template <class Real, Integer DIM>
void PtTree<Real, DIM>::GetParticleData(const std::string &data_name, Real *out, Long n, MemSpace space) const {
    if (space == MemSpace::Device) {
        p_->tree.GetParticleData(detail_deviceTree::View<Real>{out, n}, data_name);
        return;
    }
    detail_deviceTree::Scratch<Real> stage(n);
    p_->tree.GetParticleData(stage, data_name);
    gpu_tree::detail::deviceToHost(stage.data(), n, out);
}

template <class Real, Integer DIM>
void PtTree<Real, DIM>::DeleteParticleData(const std::string &data_name) {
    p_->tree.DeleteParticleData(data_name);
}

template <class Real, Integer DIM>
void PtTree<Real, DIM>::AddData(const std::string &name, Long dof, const Vector<Long> &cnt) {
    p_->tree.template AddData<Real>(name, dof, cnt);
}

template <class Real, Integer DIM>
std::span<Real> PtTree<Real, DIM>::Data(const std::string &name, Vector<Long> &cnt) {
    typename Impl::Tree::template View<Real> v;
    p_->tree.GetData(v, cnt, name);
    return {v.ptr, (std::size_t)v.size()};
}

template <class Real, Integer DIM>
void PtTree<Real, DIM>::Broadcast(const std::string &name) {
    p_->tree.Broadcast(name);
}

template <class Real, Integer DIM>
void PtTree<Real, DIM>::ReduceBroadcast(const std::string &name) {
    p_->tree.template ReduceBroadcast<Real>(name);
}

template <class Real, Integer DIM>
std::span<const Morton<DIM>> PtTree<Real, DIM>::NodeMID() const {
    const auto &v = p_->tree.GetNodeMID();
    return {detail_deviceTree::raw(v), v.size()};
}

template <class Real, Integer DIM>
std::span<const NodeAttr> PtTree<Real, DIM>::NodeAttrs() const {
    using TreeAttr = typename Impl::Tree::NodeAttr;
    static_assert(sizeof(TreeAttr) == sizeof(NodeAttr), "device_tree::NodeAttr must match the tree's node attributes.");
    const auto &v = p_->tree.GetNodeAttr();
    return {reinterpret_cast<const NodeAttr *>(detail_deviceTree::raw(v)), v.size()};
}

template <class Real, Integer DIM>
NodeLists PtTree<Real, DIM>::Lists() const {
    const auto &l = p_->tree.GetNodeLists();
    NodeLists out;
    out.parent = {detail_deviceTree::raw(l.parent), l.parent.size()};
    out.child = {detail_deviceTree::raw(l.child), l.child.size()};
    out.nbr = {detail_deviceTree::raw(l.nbr), l.nbr.size()};
    return out;
}

template <class Real, Integer DIM>
void PtTree<Real, DIM>::OwnedRange(Long &begin, Long &end) const {
    p_->tree.GetOwnedRange(begin, end);
}

template class PtTree<float, 2>;
template class PtTree<float, 3>;
template class PtTree<double, 2>;
template class PtTree<double, 3>;

} // namespace dmk::cuda::device_tree
