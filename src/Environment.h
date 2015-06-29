#ifndef __ENVIRONMENT_H__
#define __ENVIRONMENT_H__

#include "CommonIncludes.h"
#include "State.h"

template <typename Dtype>
class Environment {
public:
  Environment();
  
  shared_ptr<State<Dtype> > Observe( int action, float & reward );
  
  void SetALE( ALEInterface* ale ) {
    ale_ = ale;
  }
private:
  ALEInterface* ale_;
};

template <typename Dtype>
Environment<Dtype>::Environment() {
}

template <typename Dtype>
shared_ptr<State<Dtype> > Environment<Dtype>::Observe( int action, float & reward ) {
  static const int N = 3;
  static const float R[N] = { 1.0, 2.3, 1.6 };
  caffe::caffe_rng_gaussian( 1, R[action], 0.5f, &reward );
  static float dummy[1] = { 1.0 };
  return shared_ptr<State<Dtype> >( new State<Dtype>( dummy, 1 ) );
}

#endif // __ENVIRONMENT_H__
