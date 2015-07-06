#ifndef __ENVIRONMENT_H__
#define __ENVIRONMENT_H__

#include "CommonIncludes.h"
#include "State.h"

DECLARE_int32(clip_reward);

template <typename Dtype>
class Environment {
public:
  Environment();
  // observe reward and return the current state
  // the action is repeated for `repeat` times
  State<Dtype> Observe( int action, float* reward, int repeat );
  // stack HISTORY_SIZE frames into a single State instance and return it
  // Note : 
  // we do not reset the game on game_over here (in which case an invalid State is returned).
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
  // extract gray scale and store it into gray_scale_[]
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
  // stack 
  shared_ptr<FrameState<Dtype> > GetFrame();
private:
  enum {
    SCREEN_WIDTH = 160,
    SCREEN_HEIGHT = 210,
    CROP_WIDTH = 84,
    CROP_HEIGHT = 84,
    CROP_SIZE = 7056,
    HISTORY_SIZE = 4
  };
  ALEInterface* ale_;
  ActionVect legal_actions_;
  Dtype gray_scale_[SCREEN_WIDTH * SCREEN_HEIGHT];
  shared_ptr<FrameState<Dtype> > frames_[4];
  
  DISABLE_COPY_AND_ASSIGN( Environment );
};

#endif // __ENVIRONMENT_H__
