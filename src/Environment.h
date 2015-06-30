#ifndef __ENVIRONMENT_H__
#define __ENVIRONMENT_H__

#include "CommonIncludes.h"
#include "State.h"

template <typename Dtype>
class Environment {
public:
  Environment();
  
  State<Dtype> Observe( int action, float & reward );
  
  void SetALE( ALEInterface* ale ) {
    ale_ = ale;
    legal_actions_ = ale_->getLegalActionSet();
  }
  
  bool GameOver() {
    return ale_->game_over();
  }
  
  void ResetGame() {
    ale_->reset_game();
  }
  
  void GetFrameGrayScale( pixel_t* pixels ) {
    ColourPalette & palette = ale_->theOSystem->colourPalette();
    for ( int i = 0; i < SCREEN_HEIGHT * SCREEN_WIDTH; ++i ) {
      int r, g, b;
      palette.getRGB( pixels[i], r, g, b );
      gray_scale_[i] = (0.2126 * r + 0.7152 * g, 0.0722 * b) / 256.0;
    }
  }
  
  inline Dtype GrayScale( int x, int y ) {
    CHECK_LE( 0, x );
    if ( x >= SCREEN_HEIGHT )
      return 0;
    CHECK_LE( 0, y );
    if ( y >= SCREEN_WIDTH )
      return 0;
    return gray_scale_[x * SCREEN_WIDTH + y];
  }
  
  shared_ptr<FrameState<Dtype> > GetFrame() {
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
    
    // frame->inspect();
    
    return shared_ptr<FrameState<Dtype> >( frame );
  }
  
  // We do not reset the game on game_over here; we expect the solver do this.
  State<Dtype> GetState( bool newGame ) {
    if ( GameOver() ) {
      return State<Dtype>();
    }
    shared_ptr<FrameState<Dtype> > nowFrame = GetFrame();
    
    //nowFrame->inspect( "GetState :: nowFrame" );
    
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
  
private:
  enum {
    SCREEN_WIDTH = 160,
    SCREEN_HEIGHT = 210,
    RESCALE_HEIGHT = 84,
    RESCALE_WIDTH = 110,
    CROP_WIDTH = 84,
    CROP_HEIGHT = 84,
    CROP_SIZE = 7056,
    CROP_W_SHIFT = 0,
    CROP_H_SHIFT = 13,
    HISTORY_SIZE = 4,
    ACTION_REPEAT = 4
  };
  ALEInterface* ale_;
  ActionVect legal_actions_;
  Dtype gray_scale_[SCREEN_WIDTH * SCREEN_HEIGHT];
  shared_ptr<FrameState<Dtype> > frames_[4];
  
  DISABLE_COPY_AND_ASSIGN( Environment );
};

template <typename Dtype>
Environment<Dtype>::Environment() {
}

template <typename Dtype>
State<Dtype> Environment<Dtype>::Observe( int action, float & reward ) {
  Action a = legal_actions_[action];
  reward = 0;
  for ( int i = 0; i < ACTION_REPEAT; ++i )
    reward += ale_->act( a );
  if ( reward > 0.0 )
    reward = 1.0;
  if ( reward < 0.0 )
    reward = -1.0;
  return GetState( false );
}

#endif // __ENVIRONMENT_H__
