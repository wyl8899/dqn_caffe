#include "Environment.h"

template <typename Dtype>
Environment<Dtype>::Environment() {
}

template <typename Dtype>
State<Dtype> Environment<Dtype>::Observe( int action, float & reward, int repeat ) {
  Action a = legal_actions_[action];
  reward = 0;
  for ( int i = 0; i < repeat; ++i )
    reward += ale_->act( a );
  return GetState( false );
}

template <typename Dtype>
void Environment<Dtype>::GetFrameGrayScale( pixel_t* pixels ) {
  ColourPalette & palette = ale_->theOSystem->colourPalette();
  for ( int i = 0; i < SCREEN_HEIGHT * SCREEN_WIDTH; ++i ) {
    int r, g, b;
    palette.getRGB( pixels[i], r, g, b );
    gray_scale_[i] = (0.2126 * r + 0.7152 * g, 0.0722 * b) / 256.0;
  }
}

template <typename Dtype>
shared_ptr<FrameState<Dtype> > Environment<Dtype>::GetFrame() {
  CHECK( !GameOver() );
  const ALEScreen & screen = ale_->getScreen();
  
  CHECK_EQ( screen.height(), SCREEN_HEIGHT );
  CHECK_EQ( screen.width(), SCREEN_WIDTH );
  GetFrameGrayScale( screen.getArray() );
  
  const Dtype height_ratio = (Dtype)SCREEN_HEIGHT / RESCALE_HEIGHT;    
  const Dtype width_ratio = (Dtype)SCREEN_WIDTH / RESCALE_WIDTH;
  
  static Dtype pixels[CROP_SIZE];
  
  for ( int i = 0; i < CROP_HEIGHT; ++i ) {
    for ( int j = 0; j < CROP_WIDTH; ++j ) {
      Dtype x = (i + CROP_H_SHIFT) * height_ratio;
      Dtype y = (j + CROP_W_SHIFT) * width_ratio;
      int s = int(x), t = int(y);
      x -= s, y -= t;
      Dtype l = ( 1 - y ) * GrayScale( s, t ) + y * GrayScale( s, t + 1 );
      Dtype r = ( 1 - y ) * GrayScale( s + 1, t ) + y * GrayScale( s + 1 , t + 1 );
      pixels[i * CROP_WIDTH + j] = l * ( 1 - x ) + r * x;
    }
  }
  
  FrameState<Dtype>* frame = new FrameState<Dtype>( pixels, CROP_SIZE );
  
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
