#include "Environment.h"

template <typename Dtype>
Environment<Dtype>::Environment() {
}

template <typename Dtype>
State<Dtype> Environment<Dtype>::Observe( int action, float* reward, int repeat ) {
  Action a = legal_actions_.at( action );
  *reward = 0.0;
  for ( int i = 0; i < repeat; ++i ) {
    int r = ale_->act( a );
    // clip the reward to 1, 0, or -1 according to its sign.
    if ( FLAGS_clip_reward ) {
      if ( r > 0 )
        r = 1;
      else if ( r < 0 )
        r = -1;
    }
    *reward += r;
  }
  return GetState( false );
}

template <typename Dtype>
void Environment<Dtype>::GetFrameGrayScale( pixel_t* pixels ) {
  ColourPalette & palette = ale_->theOSystem->colourPalette();
  for ( int i = 0; i < SCREEN_HEIGHT * SCREEN_WIDTH; ++i ) {
    int r, g, b;
    palette.getRGB( pixels[i], r, g, b );
    gray_scale_[i] = (0.2126 * r + 0.7152 * g, 0.0722 * b) / 16.0;
  }
}

template <typename Dtype>
shared_ptr<FrameState<Dtype> > Environment<Dtype>::GetFrame() {
  CHECK( !GameOver() );
  const ALEScreen & screen = ale_->getScreen();
  
  CHECK_EQ( screen.height(), SCREEN_HEIGHT );
  CHECK_EQ( screen.width(), SCREEN_WIDTH );
  GetFrameGrayScale( screen.getArray() );
  
  const double xRatio = (double)SCREEN_HEIGHT / CROP_HEIGHT;    
  const double yRatio = (double)SCREEN_WIDTH / CROP_WIDTH;
  
  static Dtype pixels[CROP_SIZE];
  for ( int i = 0; i < CROP_HEIGHT; ++i ) {
    for ( int j = 0; j < CROP_WIDTH; ++j ) {
      int xL = int(i * xRatio);
      int xR = int((i + 1) * xRatio);
      int yL = int(j * yRatio);
      int yR = int((j + 1) * yRatio);
      double result = 0.0;
#define GET_WEIGHT( i, x, ratio, l, r, weight ) \
  weight = 1.0; \
  if ( x == l ) weight = x + 1 - i * ratio; \
  else if ( x == r ) weight = (i + 1) * ratio - x; \
  CHECK_LE( 0.0, weight ); \
  CHECK_LE( weight, 1.0 );
      double xWeight, yWeight;
      for ( int x = xL; x <= xR; ++x ) {
        GET_WEIGHT( i, x, xRatio, xL, xR, xWeight );
        for ( int y = yL; y <= yR; ++y ) {
          GET_WEIGHT( j, y, yRatio, yL, yR, yWeight );
          Dtype grayScale = GrayScale( x, y );
          result += (xWeight / xRatio) * (yWeight / yRatio) * grayScale;
        }
      }
#undef GET_WEIGHT
      pixels[i * CROP_WIDTH + j] = result;
    }
  }
  FrameState<Dtype>* frame = new FrameState<Dtype>( pixels, CROP_SIZE );
  //frame->inspect( "Environment::Getframe" );
  return shared_ptr<FrameState<Dtype> >( frame );
}

template <typename Dtype>
State<Dtype> Environment<Dtype>::GetState( bool newGame ) {
  if ( GameOver() ) {
    return State<Dtype>();
  }
  shared_ptr<FrameState<Dtype> > nowFrame = GetFrame();
  
  if ( newGame ) {
    for ( int i = 0; i < HISTORY_SIZE; ++i ) {
      frames_[i] = nowFrame;
    }
  } else {
    for ( int i = HISTORY_SIZE - 1; i >= 1; --i ) {
      frames_[i] = frames_[i - 1];
    }
    frames_[0] = nowFrame;
  }
  return State<Dtype>( frames_, HISTORY_SIZE );
}

INSTANTIATE_CLASS(Environment);
