#ifndef _KERNELS_H_
#define _KERNELS_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void l3d_local_kernel_directcp_cpp_(const int32_t* nd, const int32_t* ndim, const int32_t* digits,const double* rsc,const double* cen, const double* d2max, const double* sources,const int32_t* ns, const double* charge, const double* xtarg,const double* ytarg, const double* ztarg, const int32_t* nt,double* pot, const double* thresh);

void l3d_near_kernel_directcp_cpp_(const int32_t* nd, const int32_t* ndim, const int32_t* digits,const double* rsc,const double* cen, const double* bsizeinv, const double* d2max, const double* sources,const int32_t* ns, const double* charge, const double* xtarg,const double* ytarg, const double* ztarg, const int32_t* nt,double* pot);
  
void sl3d_local_kernel_directcp_cpp_(const int32_t* nd, const int32_t* ndim, const int32_t* digits,const double* rsc,const double* cen, const double* d2max, const double* sources,const int32_t* ns, const double* charge, const double* xtarg,const double* ytarg, const double* ztarg, const int32_t* nt, double* pot, const double* thresh);

void sl3d_near_kernel_directcp_cpp_(const int32_t* nd, const int32_t* ndim, const int32_t* digits,const double* rsc,const double* cen, const double* bsizeinv, const double* d2max, const double* sources,const int32_t* ns, const double* charge, const double* xtarg,const double* ytarg, const double* ztarg, const int32_t* nt,double* pot);
  
void log_local_kernel_directcp_cpp_(const int32_t* nd, const int32_t* ndim, const int32_t* digits,const double* rsc,const double* cen, const double* d2max, const double* sources,const int32_t* ns, const double* charge, const double* xtarg,const double* ytarg, const double* ztarg, const int32_t* nt,double* pot, const double* thresh);

void log_near_kernel_directcp_cpp_(const int32_t* nd, const int32_t* ndim, const int32_t* digits,const double* rsc,const double* cen, const double* bsizeinv, const double* d2max, const double* sources,const int32_t* ns, const double* charge, const double* xtarg,const double* ytarg, const double* ztarg, const int32_t* nt,double* pot);

void y3ddirectcp_cpp_(const int32_t* nd, const int32_t* digits,const double* rlambda,const double* d2max, const double* sources,const int32_t* ns, const double* charge, const double* xtarg,const double* ytarg, const double* ztarg, const int32_t* nt,double* pot, const double* thresh);

void st3d_local_kernel_directcp_cpp_(const int32_t* nd, const int32_t* ndim, const int32_t* digits,const double* rsc,const double* cen, const double* bsizeinv, const double* d2min, const double* d2max, const double* sources,const int32_t* ns, const double* charge, const double* xtarg,const double* ytarg, const double* ztarg, const int32_t* nt,double* pot);
  
void st3d_near_kernel_directcp_cpp_(const int32_t* nd, const int32_t* ndim, const int32_t* digits,const double* rsc,const double* cen, const double* bsizeinv, const double* d2min, const double* d2max, const double* sources,const int32_t* ns, const double* charge, const double* xtarg,const double* ytarg, const double* ztarg, const int32_t* nt,double* pot);

void st2d_local_kernel_directcp_cpp_(const int32_t* nd, const int32_t* ndim, const int32_t* digits,const double* rsc,const double* cen, const double* bsizeinv, const double* d2min, const double* d2max, const double* sources,const int32_t* ns, const double* charge, const double* xtarg,const double* ytarg, const double* ztarg, const int32_t* nt,double* pot);
  
void st2d_near_kernel_directcp_cpp_(const int32_t* nd, const int32_t* ndim, const int32_t* digits,const double* rsc,const double* cen, const double* bsizeinv, const double* d2min, const double* d2max, const double* sources,const int32_t* ns, const double* charge, const double* xtarg,const double* ytarg, const double* ztarg, const int32_t* nt,double* pot);

void log_local_kernel_directdp_cpp_(const int32_t* nd, const int32_t* ndim, const int32_t* digits,const double* rsc,const double* cen, const double* d2min, const double* d2max, const double* sources,const int32_t* ns, const double* dipvec, const double* xtarg,const double* ytarg, const int32_t* nt,double* pot);

#ifdef __cplusplus
}
#endif

#endif //_KERNELS_H_
