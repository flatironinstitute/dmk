"""
Compare a multi-rank MPI dump against a single-rank baseline.

Both dumps are produced by setting DMK_DEBUG_DUMP_TREE in pdmk_params.
Run this from the directory holding the .dat files.

For comparing CPU vs GPU instead, see compare_cpu_gpu.py — same idea but reads
the GPU dump from the './gpu' subdirectory.
"""

from read_dump import DMKPtTreeData, compare_coeffs


def main():
    tree_single = DMKPtTreeData(prefix='.', comm_size=1)
    tree_mpi = DMKPtTreeData(prefix='.', comm_size=2)

    for mid in tree_single.mids[0]:
        baseline = tree_single.get_proxy_coeffs_by_mid(mid)[0]
        compare_coeffs(baseline, tree_mpi.get_proxy_coeffs_by_mid(mid),
                       'proxy_up', mid=mid)

    for mid in tree_single.mids[0]:
        baseline = tree_single.get_proxy_coeffs_downward_by_mid(mid)[0]
        compare_coeffs(baseline, tree_mpi.get_proxy_coeffs_downward_by_mid(mid),
                       'proxy_down', mid=mid)


if __name__ == '__main__':
    main()
