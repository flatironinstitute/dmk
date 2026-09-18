// A thrust-free interface to SCTL's gpu_tree, which only nvcc can compile; src/cuda/device_tree.cu
// is the one unit that does. Buffers cross as pointer and length, tagged with their memory space;
// everything returned is a span over device memory.

#pragma once

#include <span>
#include <string>

#include "sctl/common.hpp" // for Long, Integer, Periodicity
#include "sctl/comm.hpp"   // for Comm
#include "sctl/comm.txx"
#include "sctl/morton.hpp" // for Morton
#include "sctl/morton.txx"
#include "sctl/vector.hpp" // for Vector
#include "sctl/vector.txx"

namespace dmk::cuda::device_tree {

using sctl::Comm;
using sctl::Integer;
using sctl::Long;
using sctl::Morton;
using sctl::Periodicity;
using sctl::Vector;

/** Which memory a caller's buffer lives in. */
enum class MemSpace { Host, Device };

/** Layout-compatible stand-in for the tree's node attributes. */
struct NodeAttr {
    unsigned char Leaf : 1, Ghost : 1;
};

/** Node index lists, one array per kind; `-1` means absent. Indices are into the node list. */
struct NodeLists {
    std::span<const Long> parent;
    std::span<const Long> child;
    std::span<const Long> nbr;
};

/** Mirrors `gpu_tree::PtTree<Real, DIM, DeviceVector>`. Every span is over device memory and stays
 * valid until the data set is reallocated, which `UpdateRefinement`, `Broadcast`, `ReduceBroadcast`
 * and the deletes do. */
template <class Real, Integer DIM>
class PtTree {
  public:
    explicit PtTree(const Comm &comm = Comm::Self());
    ~PtTree();

    PtTree(const PtTree &) = delete;
    PtTree &operator=(const PtTree &) = delete;

    /** `coord` holds `n` points, `DIM` values each. */
    void UpdateRefinement(const Real *coord, Long n, MemSpace space, Long M = 1, bool balance21 = false,
                          Periodicity periodicity = Periodicity::NONE, Integer halo_size = -1);

    void AddParticles(const std::string &name, const Real *coord, Long n, MemSpace space);

    /** `n` is the element count, `dof * Nlocal[particle_name]`; an existing set of the name is replaced. */
    void AddParticleData(const std::string &data_name, const std::string &particle_name, const Real *data, Long n,
                         MemSpace space);

    /** `dof` unwritten values per particle, to be filled through the view `Data` returns. */
    void AddParticleData(const std::string &data_name, const std::string &particle_name, Long dof);

    /** Writes `n` values in the caller's particle order; `n` is `dof` times the particles the caller
     * supplied. */
    void GetParticleData(const std::string &data_name, Real *out, Long n, MemSpace space) const;

    void DeleteParticleData(const std::string &data_name);

    /** `cnt[i] * dof` unwritten values for node i. */
    void AddData(const std::string &name, Long dof, const Vector<Long> &cnt);

    /** View of a data set's storage, with the per-node item counts. */
    std::span<Real> Data(const std::string &name, Vector<Long> &cnt);

    void Broadcast(const std::string &name);

    /** Sums onto the owner what the ranks sharing a node hold, then broadcasts the result. */
    void ReduceBroadcast(const std::string &name);

    std::span<const Morton<DIM>> NodeMID() const;
    std::span<const NodeAttr> NodeAttrs() const;
    NodeLists Lists() const;
    void OwnedRange(Long &begin, Long &end) const;

  private:
    struct Impl;
    Impl *p_ = nullptr;
};

} // namespace dmk::cuda::device_tree
