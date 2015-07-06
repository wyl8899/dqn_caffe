#include "CustomTestSolver.h"

template <typename Dtype>
State<Dtype> CustomTestSolver<Dtype>::PlayStep( State<Dtype> nowState, float & totalReward ) {
  int action;
  nowState.Feed( this->stateBlob_ );
  this->net_->ForwardTo( this->lossLayerID_ - 1 );
  action = this->GetAction();
  float reward;
  State<Dtype> state = this->environment_.Observe( action, &reward, this->frameSkip_ );
  totalReward += reward;
  return state;
}

template <typename Dtype>
void CustomTestSolver<Dtype>::Solve( const char* resume_file ) {
  LOG(INFO) << "CustomTestSolver : Testing " << this->net_->name();
  
  if (resume_file) {
    LOG(INFO) << "Restoring previous solver status from " << resume_file;
    this->Restore(resume_file);
  }
  
  State<Dtype> nowState;
  float episodeReward = 0.0, totalReward = 0.0;
  int episodeCount = 0;

  this->iter_ = 0;
  int max_iter = this->param_.max_iter();
  while (this->iter_ < max_iter) {
    const bool display = this->param_.display() 
      && this->iter_ % this->param_.display() == 0;
    this->net_->set_debug_info(display && this->param_.debug_info());
    
    // This happens when game-over occurs, or at the first iteration.
    if ( !nowState.isValid() ) {
      totalReward += episodeReward;
      if ( episodeCount ) {
        LOG(INFO) << "Episode #" << episodeCount << " ends with total score = "
          << episodeReward << ", average score = " << totalReward / episodeCount;
      }
      ++episodeCount;
      episodeReward = 0;
      this->environment_.ResetGame();
      nowState = this->environment_.GetState( true );
    }
    nowState = this->PlayStep( nowState, episodeReward );
    if (display) {
      LOG(INFO) << "Iteration " << this->iter_;
    }
    ++this->iter_;
  }
}

INSTANTIATE_CLASS(CustomTestSolver);
