#ifndef __TRANSITION_H__
#define __TRANSITION_H__

#include "CommonIncludes.h"
#include "State.h"

template <typename Dtype>
struct Transition {
  State<Dtype> state_0;
  int action;
  float reward;
  State<Dtype> state_1;
  
  Transition( State<Dtype> s0, int a, float r, State<Dtype> s1 )
    : state_0( s0 ), action( a ), reward( r ), state_1( s1 ) {
  }
};

#endif // __TRANSITION_H__
