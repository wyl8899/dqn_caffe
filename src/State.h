#ifndef __STATE_H__
#define __STATE_H__

#include "CommonIncludes.h"

template <typename Dtype>
struct State {
  Dtype* data;
  int size;
  
  State( Dtype* d, int n );
  void Feed( Blob<Dtype>* blob);
  
  DISABLE_COPY_AND_ASSIGN( State );
};

template <typename Dtype>
State<Dtype>::State( Dtype* d, int n ) {
  CHECK_GT( n, 0 );
  data = new Dtype[n];
  caffe::caffe_copy( n, d, data );
  size = n;
}

template <typename Dtype>
void State<Dtype>::Feed( Blob<Dtype>* blob ) {
  CHECK_EQ( size, blob->count() );
  Dtype* blobData = blob->mutable_cpu_data();
  caffe::caffe_copy( size, data, blobData );
}

#endif // __STATE_H__
