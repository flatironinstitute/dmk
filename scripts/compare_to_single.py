import numpy as np
import struct
import os

class MortonID:
    __slots__ = ('x', 'y', 'z', 'depth', 'dim')
    MAX_DEPTH = 62

    def __init__(self, x, y, depth, z=None):
        self.x = x
        self.y = y
        self.z = z
        self.depth = depth
        self.dim = 2 if z is None else 3

    def __hash__(self):
        return hash((self.x, self.y, self.z, self.depth))

    def __eq__(self, other):
        return (self.x == other.x and self.y == other.y
                and self.z == other.z and self.depth == other.depth)

    @property
    def coords(self):
        if self.dim == 2:
            return [self.x, self.y]
        return [self.x, self.y, self.z]


    def children(self):
        mask = 1 << (self.MAX_DEPTH - (self.depth + 1))
        coords = self.coords
        result = [MortonID(*coords, self.depth + 1) if self.dim == 2
                  else MortonID(coords[0], coords[1], self.depth + 1, coords[2])]

        for i in range(self.dim):
            new = []
            for m in result:
                mc = m.coords
                mc[i] += mask
                if self.dim == 2:
                    new.append(MortonID(mc[0], mc[1], self.depth + 1))
                else:
                    new.append(MortonID(mc[0], mc[1], self.depth + 1, mc[2]))
            result.extend(new)

        return result


    def neighbors(self, level=None, periodic=False):
        if level is None:
            level = self.depth

        maxCoord = 1 << self.MAX_DEPTH
        box_size = 1 << (self.MAX_DEPTH - level)
        coord_mask = ~(box_size - 1) & (maxCoord - 1)
        wrap_mask = maxCoord - 1

        base = [c & coord_mask for c in self.coords]
        nbrs = [base]

        for i in range(self.dim):
            new_nbrs = []
            for m in nbrs:
                for delta in (-box_size, 0, box_size):
                    mc = list(m)
                    mc[i] = (m[i] + delta) & wrap_mask
                    new_nbrs.append(mc)
            nbrs = new_nbrs

        result = []
        for mc in nbrs:
            d = level
            if not periodic:
                for i in range(self.dim):
                    orig = base[i]
                    if orig < box_size and mc[i] == (orig - box_size) & wrap_mask:
                        d = -1
                    if orig + box_size >= maxCoord and mc[i] == (orig + box_size) & wrap_mask:
                        d = -1
            if self.dim == 2:
                result.append(MortonID(mc[0], mc[1], d))
            else:
                result.append(MortonID(mc[0], mc[1], d, mc[2]))

        return result

    def __repr__(self):
        maxCoord = 1 << self.MAX_DEPTH
        a = 0.0
        s = float(1 << self.dim)
        x = self.coords
        for j in range(self.MAX_DEPTH, -1, -1):
            for i in range(self.dim - 1, -1, -1):
                s *= 0.5
                if x[i] & (1 << j):
                    a += s
        normed = [xi / maxCoord for xi in x]
        return '(' + ','.join(str(c) for c in normed) + ',{},{})'.format(self.depth, a)


def read_sctl_binary(filename, dtype):
    dtype = np.dtype(dtype)
    return np.fromfile(filename, offset=16, dtype=dtype)


def read_centers(filename, dim, real_dtype=np.float64):
    data = read_sctl_binary(filename, real_dtype)
    return data.reshape(-1, dim)


def read_params_data(filename):
    with open(filename, 'rb') as f:
        n_dim, n_order, n_pw, float_size = struct.unpack('iiii', f.read(16))
    if float_size == 4:
        real_dtype = np.float32
    else:
        real_dtype = np.float64
    return n_dim, n_order, n_pw, real_dtype


def read_morton_ids(filename, dim):
    morton_size = dim * 8 + 8

    with open(filename, 'rb') as f:
        nel = struct.unpack('Q', f.read(8))[0]
        _ = struct.unpack('Q', f.read(8))[0] # vecdim
        raw = f.read(nel * morton_size)

    results = []

    for i in range(nel):
        offset = i * morton_size
        coords = struct.unpack_from('{}Q'.format(dim), raw, offset)
        depth = struct.unpack_from('B', raw, offset + dim * 8)[0]
        if dim == 2:
            results.append(MortonID(x=coords[0], y=coords[1], depth=depth))
        else:
            results.append(MortonID(x=coords[0], y=coords[1], z=coords[2],
                                    depth=depth))

    return results


class TreeNode:
    __slots__ = ('mid', 'rank', 'idx', '_tree')

    def __init__(self, tree, mid, rank, idx):
        self._tree = tree
        self.mid = mid
        self.rank = rank
        self.idx = idx

    @property
    def is_ghost(self):
        return self._tree.is_ghost[self.rank][self.idx]

    @property
    def is_leaf(self):
        return self._tree.is_leaf[self.rank][self.idx]

    @property
    def center(self):
        return self._tree.centers[self.rank][self.idx]

    @property
    def ifpwexp(self):
        return self._tree.ifpwexp[self.rank][self.idx]

    @property
    def iftensprodeval(self):
        return self._tree.iftensprodeval[self.rank][self.idx]

    @property
    def src_counts_owned(self):
        return self._tree.src_counts_owned[self.rank][self.idx]

    @property
    def src_counts_with_halo(self):
        return self._tree.src_counts_with_halo[self.rank][self.idx]

    @property
    def proxy_up(self):
        offset = self._tree.proxy_coeffs_offsets[self.rank][self.idx]
        if offset == -1:
            return None
        return self._tree.proxy_coeffs[self.rank][offset]

    @property
    def proxy_down(self):
        offset = self._tree.proxy_coeffs_offsets_downward[self.rank][self.idx]
        if offset == -1:
            return None
        return self._tree.proxy_coeffs_downward[self.rank][offset]

    @property
    def pw_out(self):
        offset = self._tree.pw_out_offsets[self.rank][self.idx]
        if offset == -1:
            return None
        return self._tree.pw_out[self.rank][offset]

    def children(self):
        result = []
        for child_mid in self.mid.children():
            nodes = self._tree.get_nodes(child_mid)
            result.extend(nodes)
        return result

    def neighbors(self, level=None, periodic=False):
        result = []
        for nbr_mid in self.mid.neighbors(level=level, periodic=periodic):
            if nbr_mid.depth == -1:
                continue
            nodes = [a for a in self._tree.get_nodes(nbr_mid) if a.rank == self.rank]
            result.extend(nodes)
        return result

    def __repr__(self):
        return 'TreeNode(mid={}, rank={}, ghost={}, leaf={})'.format(
            self.mid, self.rank, self.is_ghost, self.is_leaf)

    def __eq__(self, other):
        return self.mid == other.mid and self.rank == other.rank

    def __hash__(self):
        return hash((self.mid, self.rank))


class DMKPtTreeData:
    """
    Read all dumped DMKPtTree data from a given communicator.

    Parameters
    ----------
    prefix : str
        Directory containing the .dat files
    comm_size : int
        Number of MPI ranks that wrote the data
    dim : int
        2 or 3
    real_dtype : np.dtype
        np.float32 or np.float64
    morton_size : int or None
        sizeof(sctl::Morton<DIM>). None for auto-detection.
    """

    def __init__(self, prefix='.', comm_size=1):
        self.prefix = prefix
        self.comm_size = comm_size
        self.n_dim, self.n_order, self.n_pw, self.real_dtype = read_params_data(os.path.join(prefix, f'dmk_params.{comm_size}.dat'))
        self.mids = self._morton_ids()
        self.centers = self._centers()
        self.is_ghost = self._is_ghost()
        self.is_leaf = self._is_leaf()
        self.ifpwexp = self._ifpwexp()
        self.iftensprodeval = self._iftensprodeval()
        self.pw_out = self._pw_out()
        self.pw_out_offsets = self._pw_out_offsets()
        self.proxy_coeffs = self._proxy_coeffs()
        self.proxy_coeffs_offsets = self._proxy_coeffs_offsets()
        self.proxy_coeffs_downward = self._proxy_coeffs_downward()
        self.proxy_coeffs_offsets_downward = self._proxy_coeffs_offsets_downward()
        self.src_counts_with_halo = self._src_counts_with_halo()
        self.src_counts_owned = self._src_counts_owned()

        self.mid_map = [{} for _ in range(self.comm_size)]
        for rank in range(self.comm_size):
            for idx, mid in enumerate(self.mids[rank]):
               self.mid_map[rank][mid] = idx

    def get_nodes(self, mid):
        """Return list of TreeNode for every rank that holds this mid."""
        result = []
        for rank in range(self.comm_size):
            if mid in self.mid_map[rank]:
                idx = self.mid_map[rank][mid]
                result.append(TreeNode(self, mid, rank, idx))
        return result

    def get_node(self, mid, rank=None):
        """Return a single TreeNode. If rank is None, return first match."""
        if rank is not None:
            if mid in self.mid_map[rank]:
                return TreeNode(self, mid, rank, self.mid_map[rank][mid])
            return None
        for r in range(self.comm_size):
            if mid in self.mid_map[r]:
                return TreeNode(self, mid, r, self.mid_map[r][mid])
        return None

    def all_nodes(self, rank=None):
        """Iterate over all nodes, optionally filtered by rank."""
        if rank is not None:
            for idx, mid in enumerate(self.mids[rank]):
                yield TreeNode(self, mid, rank, idx)
        else:
            for r in range(self.comm_size):
                for idx, mid in enumerate(self.mids[r]):
                    yield TreeNode(self, mid, r, idx)

    def get_children_by_mid(self, mid):
        children_mids = mid.children()
        res = []
        for rank in range(self.comm_size):
            local_res = []
            for child_mid in children_mids:
                if child_mid in self.mid_map[rank]:
                    local_res.append(child_mid)
            res.append(local_res)
        return res
                

    def get_mid_by_string(self, mid_str):
        for rank in range(self.comm_size):
            for mid in self.mids[rank]:
                if str(mid) == mid_str:
                    return mid
        return None

    def get_indices_by_mid(self, mid):
        indices = []
        for rank in range(self.comm_size):
            if mid in self.mid_map[rank]:
                indices.append((rank, self.mid_map[rank][mid]))
            else:
                indices.append((rank, None))
        return indices

    def get_proxy_coeffs_by_mid(self, mid):
        indices = self.get_indices_by_mid(mid)
        coeffs = []
        for rank, idx in indices:
            if idx is None:
                coeffs.append(None)
                continue
            offset = self.proxy_coeffs_offsets[rank][idx]
            if offset != -1:
                coeffs.append(self.proxy_coeffs[rank][offset])
            else:
                coeffs.append(None)
        return coeffs

    def get_pw_out_by_mid(self, mid):
        indices = self.get_indices_by_mid(mid)
        pw_outs = []
        for rank, idx in indices:
            if idx is None:
                pw_outs.append(None)
                continue
            offset = self.pw_out_offsets[rank][idx]
            if offset != -1:
                pw_outs.append(self.pw_out[rank][offset])
            else:
                pw_outs.append(None)
        return pw_outs

    def get_proxy_coeffs_downward_by_mid(self, mid):
        indices = self.get_indices_by_mid(mid)
        coeffs = []
        for rank, idx in indices:
            if idx is None:
                coeffs.append(None)
                continue
            offset = self.proxy_coeffs_offsets_downward[rank][idx]
            if offset != -1:
                coeffs.append(self.proxy_coeffs_downward[rank][offset])
            else:
                coeffs.append(None)
        return coeffs

    def _filename(self, name, rank):
        return os.path.join(
            self.prefix,
            '{}.{}.{}.dat'.format(name, self.comm_size, rank))

    def _read_all_ranks(self, name, dtype):
        """Read and concatenate data from all ranks."""
        arrays = []
        for rank in range(self.comm_size):
            arrays.append(read_sctl_binary(self._filename(name, rank), dtype))
        return arrays

    @property
    def proxy_shape(self):
        if self.n_dim == 2:
            return (self.n_order, self.n_order)
        else:
            return (self.n_order, self.n_order, self.n_order)
    
    @property
    def pw_shape(self):
        if self.n_dim == 2:
            return (self.n_pw, (self.n_pw + 1) // 2)
        else:
            return (self.n_pw, self.n_pw, (self.n_pw + 1) // 2)

    def _centers(self):
        return [out.reshape(-1, self.n_dim) for out in self._read_all_ranks('dmk_centers', self.real_dtype)]

    def _is_ghost(self):
        return self._read_all_ranks('dmk_is_ghost', np.bool_)

    def _is_leaf(self):
        return self._read_all_ranks('dmk_is_leaf', np.bool_)

    def _ifpwexp(self):
        return self._read_all_ranks('dmk_ifpwexp', np.bool_)

    def _iftensprodeval(self):
        return self._read_all_ranks('dmk_iftensprodeval', np.bool_)

    def _pw_out(self):
        return [out.reshape(-1, *self.pw_shape) for out in self._read_all_ranks('dmk_pw_out', self.real_dtype)]

    def _pw_out_offsets(self):
        pw_size = np.prod(self.pw_shape)
        raw = self._read_all_ranks('dmk_pw_out_offsets', np.int64)
        for rank in range(self.comm_size):
            raw[rank] = np.array([int(el // pw_size) if el != -1 else int(el) for el in raw[rank]])
        return raw

    def _proxy_coeffs(self):
        return [out.reshape(-1, *self.proxy_shape) for out in self._read_all_ranks('dmk_proxy_coeffs', self.real_dtype)]

    def _proxy_coeffs_offsets(self):
        proxy_size = np.prod(self.proxy_shape)
        raw = self._read_all_ranks('dmk_proxy_coeffs_offsets', np.int64)
        for rank in range(self.comm_size):
            raw[rank] = np.array([int(el // proxy_size) if el != -1 else int(el) for el in raw[rank]])
        return raw

    def _proxy_coeffs_downward(self):
        return [out.reshape(-1, *self.proxy_shape) for out in self._read_all_ranks('dmk_proxy_coeffs_downward', self.real_dtype)]

    def _proxy_coeffs_offsets_downward(self):
        proxy_size = np.prod(self.proxy_shape)
        raw = self._read_all_ranks('dmk_proxy_coeffs_offsets_downward', np.int64)
        for rank in range(self.comm_size):
            raw[rank] = np.array([int(el // proxy_size) if el != -1 else int(el) for el in raw[rank]])
        return raw

    def _pw_out_downward(self):
        return [out.reshape(-1, *self.pw_shape) for out in self._read_all_ranks('dmk_pw_out', self.real_dtype)]

    def _pw_out_offsets_downward(self):
        pw_size = np.prod(self.pw_shape)
        raw = self._read_all_ranks('dmk_pw_out_offsets', np.int64)
        for rank in range(self.comm_size):
            raw[rank] = np.array([int(el // pw_size) if el != -1 else int(el) for el in raw[rank]])
        return raw

    def _src_counts_owned(self):
        return self._read_all_ranks('dmk_src_counts_owned', np.int32)

    def _src_counts_with_halo(self):
        return self._read_all_ranks('dmk_src_counts_with_halo', np.int32)

    def _morton_ids(self):
        """List of MortonID namedtuples from all ranks."""
        results = []
        for rank in range(self.comm_size):
            results.append(
                read_morton_ids(
                    self._filename('dmk_morton_ids', rank),
                    self.n_dim))
        return results

    def __repr__(self):
        return f"DMKPtTreeData(prefix='{self.prefix}', comm_size={self.comm_size}, dim={self.n_dim}, n_order={self.n_order}, n_pw={self.n_pw}, real_dtype={self.real_dtype})"


def compare_coeffs(coeffs_baseline, coeffs_mpi, tag):
    if coeffs_baseline is None:
        return

    for coeffs in coeffs_mpi:
        if coeffs is not None:
            mask = coeffs_baseline != 0
            if not np.any(mask):
                if np.any(coeffs[mask]):
                    print(mid, "nonzero coeffs where baseline is zero")
                continue

            maxerr = np.max(np.abs(1 - coeffs[mask]/coeffs_baseline[mask]))
            if maxerr > 1e-7:
                print(tag, mid, maxerr)
    

tree_single = DMKPtTreeData(comm_size=1)
tree_mpi = DMKPtTreeData(comm_size=2)

for mid in tree_single.mids[0]:
    coeffs_baseline = tree_single.get_proxy_coeffs_by_mid(mid)[0]
    coeffs_mpi = tree_mpi.get_proxy_coeffs_by_mid(mid)

    compare_coeffs(coeffs_baseline, coeffs_mpi, "proxy_up")

    
for mid in tree_single.mids[0]:
    coeffs_baseline = tree_single.get_proxy_coeffs_downward_by_mid(mid)[0]
    coeffs_mpi = tree_mpi.get_proxy_coeffs_downward_by_mid(mid)

    compare_coeffs(coeffs_baseline, coeffs_mpi, "proxy_down")
