#!/usr/bin/env python3

import numpy as np

norder = 38
ndim = 2
nel = norder**ndim


lpaddr = np.fromfile('lpaddr_fort.bin', dtype=np.int64)
lpaddr = lpaddr.reshape((lpaddr.size//2, 2)) - 1
nboxes = lpaddr.shape[0]

ifpwexp = np.fromfile('dmk_ifpwexp.1.0.dat', offset=16, dtype=np.int32).astype(np.bool_)
iftensprodeval = np.fromfile('dmk_iftensprodeval.1.0.dat', offset=16, dtype=np.int32).astype(np.bool_)

ifpwexp_fort = np.fromfile('form_pw_fort.bin', dtype=np.int32).astype(np.bool_)
form_tp_fort = np.fromfile('form_tp_fort.bin', dtype=np.int32).astype(np.bool_)
iftensprodeval_fort = np.fromfile('eval_tp_fort.bin', dtype=np.int32).astype(np.bool_)

centers = np.fromfile('dmk_centers.1.0.dat', offset=16, dtype=np.float64).reshape(-1, ndim)
centers_fort = np.fromfile('centers_fort.bin', dtype=np.float64).reshape(-1, ndim)

proxy_coeffs = np.fromfile('dmk_proxy_coeffs.1.0.dat', offset=16, dtype=np.float64).reshape(-1, nel)
proxy_coeffs_offsets = np.fromfile('dmk_proxy_coeffs_offsets.1.0.dat', offset=16, dtype=np.int64)
proxy_coeffs_downward = np.fromfile('dmk_proxy_coeffs_downward.1.0.dat', offset=16, dtype=np.float64).reshape(-1, nel)
proxy_coeffs_downward_offsets = np.fromfile('dmk_proxy_coeffs_offsets_downward.1.0.dat', offset=16, dtype=np.int64)

proxy_coeffs_all_fort = np.fromfile('proxy_coeffs_all_fort.bin', dtype=np.float64)
# proxy_coeffs_all_fort = proxy_coeffs_all_fort.reshape((proxy_coeffs_all_fort.size//nel, nel))

cppboxes = []
for fortbox in range(nboxes):
    cppbox = np.where((np.isclose(centers,centers_fort[fortbox]).all(axis=1)))[0][0]
    cppboxes.append(int(cppbox))

assert np.array_equal(ifpwexp[cppboxes], ifpwexp_fort)
assert np.array_equal(iftensprodeval[cppboxes], iftensprodeval_fort)

for fortbox in range(nboxes):
    cppbox = cppboxes[fortbox]
    
    offset = lpaddr[fortbox,0]
    proxy_box = proxy_coeffs_offsets[cppbox] // nel
    if proxy_box >= 0 and offset >= 0:
        up_same = np.allclose(proxy_coeffs[proxy_box], proxy_coeffs_all_fort[offset:offset+nel])
    else:
        up_same = "NA"

    if up_same is False:
        print("up: ", cppbox, fortbox, proxy_coeffs[proxy_box][0], proxy_coeffs_all_fort[offset])

    offset = lpaddr[fortbox,1]
    proxy_box = proxy_coeffs_downward_offsets[cppbox] // nel    
    if proxy_box >= 0 and offset >= 0:
        down_same = np.allclose(proxy_coeffs_downward[proxy_box], proxy_coeffs_all_fort[offset:offset+nel])
    else:
        down_same = "NA"

    if down_same is False:
        print("down:", cppbox, fortbox, proxy_coeffs_downward[proxy_box][0], proxy_coeffs_all_fort[offset])

#     if not (up_same == "NA" and down_same == "NA"):
#         if down_same is False:
#             print(cppbox, fortbox, up_same, down_same)
