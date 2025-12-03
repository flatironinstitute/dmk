#include <kernels7.h>
#include <template-kernels7.hpp>
#define VECDIM 4

#ifdef __AVX512F__
#undef VECDIM
#define VECDIM 8
#endif

#ifdef __cplusplus
extern "C" {
#endif
  // for stokesdmk7.f

void st3d_local_kernel_directcp_cpp_(const int32_t* nd, const int32_t* ndim, const int32_t* digits,const double* rsc,const double* cen, const double* bsizeinv, const double *d2min, const double* d2max, const double* sources,const int32_t* ns, const double* charge, const double* xtarg,const double* ytarg,const double* ztarg,const int32_t* nt, double* pot){
  st3d_local_kernel_directcp_vec_cpp<double, VECDIM>(nd, ndim, digits,rsc, cen, bsizeinv, d2min, d2max, sources, ns, charge,xtarg, ytarg, ztarg, nt, pot);
}

void st2d_local_kernel_directcp_cpp_(const int32_t* nd, const int32_t* ndim, const int32_t* digits,const double* rsc,const double* cen, const double* bsizeinv, const double *d2min, const double* d2max, const double* sources,const int32_t* ns, const double* charge, const double* xtarg,const double* ytarg,const double* ztarg,const int32_t* nt, double* pot){
  st2d_local_kernel_directcp_vec_cpp<double, VECDIM>(nd, ndim, digits,rsc, cen, bsizeinv, d2min, d2max, sources, ns, charge,xtarg, ytarg, ztarg, nt, pot);
}

void st3d_local_kernel_directdp_cpp_(const int32_t* nd, const int32_t* ndim, const int32_t* digits,const double* rsc,const double* cen, const double* bsizeinv, const double *d2min, const double* d2max, const double* sources,const int32_t* ns, const double* strslet, const double* strsvec, const double* xtarg,const double* ytarg,const double* ztarg,const int32_t* nt, double* pot){
  st3d_local_kernel_directdp_vec_cpp<double, VECDIM>(nd, ndim, digits,rsc, cen, bsizeinv, d2min, d2max, sources, ns, strslet, strsvec, xtarg, ytarg, ztarg, nt, pot);
}

void st2d_local_kernel_directdp_cpp_(const int32_t* nd, const int32_t* ndim, const int32_t* digits,const double* rsc,const double* cen, const double* bsizeinv, const double *d2min, const double* d2max, const double* sources,const int32_t* ns, const double* strslet, const double* strsvec, const double* xtarg,const double* ytarg,const double* ztarg,const int32_t* nt, double* pot){
	st2d_local_kernel_directdp_vec_cpp<double, VECDIM>(nd, ndim, digits,rsc, cen, bsizeinv, d2min, d2max, sources, ns, strslet, strsvec, xtarg, ytarg, ztarg, nt, pot);
}
  
#ifdef __cplusplus
}
#endif

