#ifndef __STATE_H__
#define __STATE_H__

#include "CommonIncludes.h"

template <typename Dtype>
struct FrameState {
  enum {
    FRAME_WIDTH = 84,
    FRAME_HEIGHT = 84,
    FRAME_SIZE = 7056
  };
  Dtype* data;
  
  FrameState( Dtype* d, int n ) {
    CHECK_EQ( n, FRAME_SIZE );
    data = new Dtype[n];
    caffe::caffe_copy( n, d, data );
  }
  
  void inspect( string text = "" ) {
    Dtype* pixels = data;
    cout << text << " --> inspect() : pixels = " << pixels << endl;
    for ( int i = 0; i < FRAME_HEIGHT; ++i ) {
      for ( int j = 0; j < FRAME_WIDTH; ++j ) {
        cout << int(pixels[i * FRAME_WIDTH + j] * 256 > 0.5);
      }
      cout << endl;
    }
    cout << text << " <--\n";
  }
  
  DISABLE_COPY_AND_ASSIGN( FrameState );
};

template <typename Dtype>
struct State {
  typedef FrameState<Dtype> Frame;
  typedef shared_ptr<Frame> FramePtr;
  enum {
    HISTORY_SIZE = 4
  };
  FramePtr frames[HISTORY_SIZE];
  
  State() {
    for ( int i = 0; i < HISTORY_SIZE; ++i ) {
      frames[i] = FramePtr( (Frame*)NULL );
      CHECK( !frames[i] );
    }
    CHECK( !isValid() );
  }
  
  State( FramePtr* f, int n ) {
    CHECK_EQ( n, HISTORY_SIZE );
    for ( int i = 0; i < HISTORY_SIZE; ++i )
      frames[i] = f[i];
    CHECK( isValid() );
  }
  
  bool isValid() {
    for ( int i = 0; i < HISTORY_SIZE; ++i )
      if ( !frames[i] )
        return false;
    return true;
  }
  
  void inspect( string text ) {
    for ( int i = 0; i < HISTORY_SIZE; ++i )
      frames[i]->inspect( text + " --> State::inspect() " );
    cout << text << " <--\n";
  }
  
  void Feed( Blob<Dtype>* blob ) {
    CHECK_EQ( HISTORY_SIZE * Frame::FRAME_SIZE, blob->count() );
    Dtype* blobData = blob->mutable_cpu_data();
    for ( int i = 0; i < HISTORY_SIZE; ++i ) {
      caffe::caffe_copy( Frame::FRAME_SIZE,
        frames[i]->data, blobData + i * Frame::FRAME_SIZE );
    }
  }
};

#endif // __STATE_H__
