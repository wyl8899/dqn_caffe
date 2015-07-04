#ifndef __ENVIRONMENT_H__
#define __ENVIRONMENT_H__

#include "CommonIncludes.h"
#include "State.h"

template <typename Dtype>
class Environment {
public:
  Environment();
  State<Dtype> Observe( int action, float & reward, int repeat = FRAME_SKIP );
  // We do not reset the game on game_over here (in which case an invalid State is returned).
  // Rather, we expect the solver do this, making it able to know when an episode ends.
  // newGame acts as a signal to reset the frame history.
  // When newGame is true, we copy the newly observed frame to fill the whole history.
  State<Dtype> GetState( bool newGame );
  inline void SetALE( ALEInterface* ale ) {
    ale_ = ale;
    legal_actions_ = ale_->getMinimalActionSet();
  }
  inline int GetLegalActionCount() {
    return legal_actions_.size();
  }
  inline bool GameOver() {
    return ale_->game_over();
  }
  inline void ResetGame() {
    ale_->reset_game();
  }
protected:
  void GetFrameGrayScale( pixel_t* pixels );
  inline Dtype GrayScale( int x, int y ) {
    CHECK_LE( 0, x );
    if ( x >= SCREEN_HEIGHT )
      return 0;
    CHECK_LE( 0, y );
    if ( y >= SCREEN_WIDTH )
      return 0;
    return gray_scale_[x * SCREEN_WIDTH + y];
  }
  shared_ptr<FrameState<Dtype> > GetFrame();
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
    FRAME_SKIP = 4
  };
  ALEInterface* ale_;
  ActionVect legal_actions_;
  Dtype gray_scale_[SCREEN_WIDTH * SCREEN_HEIGHT];
  shared_ptr<FrameState<Dtype> > frames_[4];
  
  DISABLE_COPY_AND_ASSIGN( Environment );
};

#endif // __ENVIRONMENT_H__
