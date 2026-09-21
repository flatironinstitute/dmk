// The one translation unit nvcc compiles; defines the interface declared in dmk/cuda/device_tree.hpp.

#include "sctl/tree.hpp"
#include "sctl/tree.txx"
#include "sctl/experimental/gpu-tree.hpp"
#include "dmk/cuda/device_tree.hpp"
#include "dmk/cuda/device_vector.hpp"

#include <thrust/copy.h>
#include <thrust/device_ptr.h>

namespace dmk::cuda::device_tree {

namespace detail_deviceTree {

/** The caller's buffer as a device-side range. */
template <class T>
DeviceVector<T> toDevice(const T *src, Long n, MemSpace space) {
    if (space == MemSpace::Host)
        return DeviceVector<T>(src, src + n);
    const thrust::device_ptr<const T> p(src);
    return DeviceVector<T>(p, p + n);
}

/** The tree's buffer into the caller's. */
template <class T>
void fromDevice(const DeviceVector<T> &src, T *dst, MemSpace space) {
    if (space == MemSpace::Host)
        thrust::copy(src.begin(), src.end(), dst);
    else
        thrust::copy(src.begin(), src.end(), thrust::device_ptr<T>(dst));
}

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
    const auto x = detail_deviceTree::toDevice(coord, n * DIM, space);
    p_->tree.UpdateRefinement(x, M, balance21, periodicity, halo_size);
}

template <class Real, Integer DIM>
void PtTree<Real, DIM>::AddParticles(const std::string &name, const Real *coord, Long n, MemSpace space) {
    const auto x = detail_deviceTree::toDevice(coord, n * DIM, space);
    p_->tree.AddParticles(name, x);
}

template <class Real, Integer DIM>
void PtTree<Real, DIM>::AddParticleData(const std::string &data_name, const std::string &particle_name,
                                        const Real *data, Long n, MemSpace space) {
    const auto v = detail_deviceTree::toDevice(data, n, space);
    p_->tree.AddParticleData(data_name, particle_name, v);
}

template <class Real, Integer DIM>
void PtTree<Real, DIM>::AddParticleData(const std::string &data_name, const std::string &particle_name, Long dof) {
    p_->tree.AddParticleData(data_name, particle_name, dof);
}

template <class Real, Integer DIM>
void PtTree<Real, DIM>::GetParticleData(const std::string &data_name, Real *out, Long n, MemSpace space) const {
    DeviceVector<Real> v;
    p_->tree.GetParticleData(v, data_name);
    SCTL_ASSERT_MSG((Long)v.size() == n, "device_tree::PtTree::GetParticleData: n does not match the data.");
    detail_deviceTree::fromDevice(v, out, space);
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
