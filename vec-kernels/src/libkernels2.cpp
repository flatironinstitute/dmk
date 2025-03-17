#include <kernels.h>
#include <template-kernels2.hpp>
#define VECDIM 4

#ifdef __AVX512F__
#undef VECDIM
#define VECDIM 8
#endif

#ifdef __cplusplus

template void l3d_local_kernel_directcp_vec_cpp<float, VECDIM>(
    const int32_t *nd, const int32_t *ndim, const int32_t *digits, const float *rsc, const float *cen,
    const float *d2max, const float *sources, const int32_t *ns, const float *charge, const float *xtarg,
    const float *ytarg, const float *ztarg, const int32_t *nt, float *pot, const float *thresh);

extern "C" {
#endif

void l3d_local_kernel_directcp_cpp_(const int32_t* nd, const int32_t* ndim, const int32_t* digits,const double* rsc,const double* cen, const double* d2max, const double* sources,const int32_t* ns, const double* charge, const double* xtarg,const double* ytarg,const double* ztarg,const int32_t* nt, double* pot, const double* thresh){
  l3d_local_kernel_directcp_vec_cpp<double, VECDIM>(nd, ndim, digits,rsc, cen, d2max, sources, ns, charge,xtarg, ytarg, ztarg, nt, pot, thresh);
}

void l3d_near_kernel_directcp_cpp_(const int32_t* nd, const int32_t* ndim, const int32_t* digits,const double* rsc,const double* cen, const double* bsizeinv, const double* d2max, const double* sources,const int32_t* ns, const double* charge, const double* xtarg,const double* ytarg,const double* ztarg,const int32_t* nt, double* pot){
  l3d_near_kernel_directcp_vec_cpp<double, VECDIM>(nd, ndim, digits,rsc, cen, bsizeinv, d2max, sources, ns, charge,xtarg, ytarg, ztarg, nt, pot);
}
  
void sl3d_local_kernel_directcp_cpp_(const int32_t* nd, const int32_t* ndim, const int32_t* digits,const double* rsc,const double* cen, const double* d2max, const double* sources,const int32_t* ns, const double* charge, const double* xtarg,const double* ytarg,const double* ztarg,const int32_t* nt, double* pot, const double* thresh){
  sl3d_local_kernel_directcp_vec_cpp<double, VECDIM>(nd, ndim, digits,rsc, cen, d2max, sources, ns, charge,xtarg, ytarg, ztarg, nt, pot, thresh);
}

void sl3d_near_kernel_directcp_cpp_(const int32_t* nd, const int32_t* ndim, const int32_t* digits,const double* rsc,const double* cen, const double* bsizeinv, const double* d2max, const double* sources,const int32_t* ns, const double* charge, const double* xtarg,const double* ytarg,const double* ztarg,const int32_t* nt, double* pot){
  sl3d_near_kernel_directcp_vec_cpp<double, VECDIM>(nd, ndim, digits,rsc, cen, bsizeinv, d2max, sources, ns, charge,xtarg, ytarg, ztarg, nt, pot);
}

void log_local_kernel_directcp_cpp_(const int32_t* nd, const int32_t* ndim, const int32_t* digits,const double* rsc,const double* cen, const double* d2max, const double* sources,const int32_t* ns, const double* charge, const double* xtarg,const double* ytarg,const double* ztarg,const int32_t* nt, double* pot, const double* thresh){
  log_local_kernel_directcp_vec_cpp<double, VECDIM>(nd, ndim, digits,rsc, cen, d2max, sources, ns, charge,xtarg, ytarg, ztarg, nt, pot, thresh);
}

void log_near_kernel_directcp_cpp_(const int32_t* nd, const int32_t* ndim, const int32_t* digits,const double* rsc,const double* cen, const double* bsizeinv, const double* d2max, const double* sources,const int32_t* ns, const double* charge, const double* xtarg,const double* ytarg,const double* ztarg,const int32_t* nt, double* pot){
  log_near_kernel_directcp_vec_cpp<double, VECDIM>(nd, ndim, digits,rsc, cen, bsizeinv, d2max, sources, ns, charge,xtarg, ytarg, ztarg, nt, pot);
}

void y3ddirectcp_cpp_(const int32_t* nd, const int32_t* digits,const double* rlambda,const double* d2max, const double* sources,const int32_t* ns, const double* charge, const double* xtarg,const double* ytarg,const double* ztarg,const int32_t* nt, double* pot, const double* thresh){
  y3ddirectcp_vec_cpp<double, VECDIM>(nd, digits,rlambda, d2max, sources, ns, charge,xtarg, ytarg, ztarg, nt, pot, thresh);
}

void st3d_local_kernel_directcp_cpp_(const int32_t* nd, const int32_t* ndim, const int32_t* digits,const double* rsc,const double* cen, const double* bsizeinv, const double *d2min, const double* d2max, const double* sources,const int32_t* ns, const double* charge, const double* xtarg,const double* ytarg,const double* ztarg,const int32_t* nt, double* pot){
  st3d_local_kernel_directcp_vec_cpp<double, VECDIM>(nd, ndim, digits,rsc, cen, bsizeinv, d2min, d2max, sources, ns, charge,xtarg, ytarg, ztarg, nt, pot);
}

void st3d_near_kernel_directcp_cpp_(const int32_t* nd, const int32_t* ndim, const int32_t* digits,const double* rsc,const double* cen, const double* bsizeinv, const double* d2min, const double* d2max, const double* sources,const int32_t* ns, const double* charge, const double* xtarg,const double* ytarg,const double* ztarg,const int32_t* nt, double* pot){
  st3d_near_kernel_directcp_vec_cpp<double, VECDIM>(nd, ndim, digits,rsc, cen, bsizeinv, d2min, d2max, sources, ns, charge,xtarg, ytarg, ztarg, nt, pot);
}

void st2d_local_kernel_directcp_cpp_(const int32_t* nd, const int32_t* ndim, const int32_t* digits,const double* rsc,const double* cen, const double* bsizeinv, const double *d2min, const double* d2max, const double* sources,const int32_t* ns, const double* charge, const double* xtarg,const double* ytarg,const double* ztarg,const int32_t* nt, double* pot){
  st2d_local_kernel_directcp_vec_cpp<double, VECDIM>(nd, ndim, digits,rsc, cen, bsizeinv, d2min, d2max, sources, ns, charge,xtarg, ytarg, ztarg, nt, pot);
}

void st2d_near_kernel_directcp_cpp_(const int32_t* nd, const int32_t* ndim, const int32_t* digits,const double* rsc,const double* cen, const double* bsizeinv, const double* d2min, const double* d2max, const double* sources,const int32_t* ns, const double* charge, const double* xtarg,const double* ytarg,const double* ztarg,const int32_t* nt, double* pot){
  st2d_near_kernel_directcp_vec_cpp<double, VECDIM>(nd, ndim, digits,rsc, cen, bsizeinv, d2min, d2max, sources, ns, charge,xtarg, ytarg, ztarg, nt, pot);
}

void log_local_kernel_directdp_cpp_(const int32_t* nd, const int32_t* ndim, const int32_t* digits,const double* rsc,const double* cen, const double *d2min, const double* d2max, const double* sources,const int32_t* ns, const double* dipvec, const double* xtarg,const double* ytarg,const int32_t* nt, double* pot){
  log_local_kernel_directdp_vec_cpp<double, VECDIM>(nd, ndim, digits,rsc, cen, d2min, d2max, sources, ns, dipvec, xtarg, ytarg, nt, pot);
}


#ifdef __cplusplus
}
#endif

