#ifndef __TRANSITION_H__
#define __TRANSITION_H__

#include "CommonIncludes.h"
#include "State.h"

template <typename Dtype>
struct Transition {
  typedef shared_ptr<State<Dtype> > StatePtr;
  
  StatePtr state_0;
  int action;
  float reward;
  StatePtr state_1;
  
  Transition( StatePtr s0, int a, float r, StatePtr s1 );
  
  void FeedState( int i, Blob<Dtype>* blob );
};

template <typename Dtype>
Transition<Dtype>::Transition( StatePtr s0, int a, float r, StatePtr s1 )
  : state_0( s0 ), action( a ), reward( r ), state_1( s1 ) {
}

template <typename Dtype>
void Transition<Dtype>::FeedState( int i, Blob<Dtype>* blob ) {
  if( i == 0 )
    state_0->Feed( blob );
  else if ( i == 1 )
    state_1->Feed( blob );
  else
    LOG(FATAL) << "No such state";
}

#endif // __TRANSITION_H__
