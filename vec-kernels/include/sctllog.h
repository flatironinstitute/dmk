#ifndef _SCTLLOG_H_
#define _SCTLLOG_H_

#include "sctl.hpp"
#include "vectorclass.h"
#include "vectormath_exp.h"

namespace sctl {

#ifdef __SSE4_2__
inline sctl::Vec<float,4> veclog(const sctl::Vec<float,4>& v) {
  Vec4f v_(v.get().v);
  return (sctl::Vec<float,4>::VData)(__m128)log(v_);
}
#endif
#ifdef __AVX__
inline sctl::Vec<float,8> veclog(const sctl::Vec<float,8>& v) {
  Vec8f v_(v.get().v);
  return (sctl::Vec<float,8>::VData)(__m256)log(v_);
}
#endif
#ifdef __AVX512F__
inline sctl::Vec<float,16> veclog(const sctl::Vec<float,16>& v) {
  Vec16f v_(v.get().v);
  return (sctl::Vec<float,16>::VData)(__m512)log(v_);
}
#endif


#ifdef __SSE4_2__
inline sctl::Vec<double,2> veclog(const sctl::Vec<double,2>& v) {
  Vec2d v_(v.get().v);
  return (sctl::Vec<double,2>::VData)(__m128d)log(v_);
}
#endif
#ifdef __AVX__
inline sctl::Vec<double,4> veclog(const sctl::Vec<double,4>& v) {
  Vec4d v_(v.get().v);
  return (sctl::Vec<double,4>::VData)(__m256d)log(v_);
}
#endif
#ifdef __AVX512F__
inline sctl::Vec<double,8> veclog(const sctl::Vec<double,8>& v) {
  Vec8d v_(v.get().v);
  return (sctl::Vec<double,8>::VData)(__m512d)log(v_);
}
#endif

}

#endif
