#ifndef __CUSTOM_SOLVER_H__
#define __CUSTOM_SOLVER_H__

#include "CommonIncludes.h"
#include "State.h"
#include "Environment.h"
#include "Transition.h"
#include "ExpHistory.h"

#define FIND_BLOB(name) \
  if ( blobs[i] == #name ) { \
    name##Blob_ = this->net_->blobs()[i].get(); \
    LOG(INFO) << "Blob " << #name << " found. "; \
  }

template <typename Dtype>
class CustomSolver : public SGDSolver<Dtype> {
public:
  explicit CustomSolver( const SolverParameter& param )
    : SGDSolver<Dtype>( param ), history_( HISTORY_SIZE ) {
    const vector<string> & layers = this->net_->layer_names();
    for ( int i = 0; i < layers.size(); ++i ) {
      LOG(INFO) << "Layer #" << i << ": " << layers[i];
      if ( layers[i] == "loss" )
        lossLayerID_ = i;
    }
    const vector<string> & blobs = this->net_->blob_names();
    for ( int i = 0; i < blobs.size(); ++i ) {
      LOG(INFO) << "Blob #" << i << ": " << blobs[i];
      FIND_BLOB(state);
      FIND_BLOB(reward);
      FIND_BLOB(pred);
      FIND_BLOB(action);
    }
  }
  void Step( int );
  
  void SetALE( ALEInterface* ale ) {
    environment_.SetALE( ale );
  }
  
private:
  enum {
    NUMBER_OF_LEGAL_ACTIONS = 18,
    REPLAY_START_SIZE = 5000,
    HISTORY_SIZE = 10000
  };
  
  ExpHistory<Dtype> history_;
  Environment<Dtype> environment_;
  
  int lossLayerID_;
  Blob<Dtype> *stateBlob_;
  Blob<Dtype> *rewardBlob_;
  Blob<Dtype> *predBlob_;
  Blob<Dtype> *actionBlob_;
  
  void FeedState();
  int GetActionFromNet();
  int GetAction( float epsilon );
  void FeedReward( int action, float reward );
  
  shared_ptr<State<Dtype> > PlayStep( shared_ptr<State<Dtype> > state, float & totalReward );
  float TrainStep();
  
  float GetEpsilon() {
    const float INITIAL_EPSILON = 1.0;
    const float FINAL_EPSILON = 0.1;
    const int FINAL_FRAME = 1000000; 
    if ( this->iter_ > FINAL_FRAME )
      return FINAL_EPSILON;
    else
      return INITIAL_EPSILON - 
        (INITIAL_EPSILON - FINAL_EPSILON) * ((float)this->iter_ / FINAL_FRAME);
  }
  
  int GetActionFromRandom() {
    return rand() % NUMBER_OF_LEGAL_ACTIONS;
  }
};

template <typename Dtype>
void CustomSolver<Dtype>::FeedState() {
  const vector<Blob<Dtype>*> & inputs = this->net_->input_blobs();
  Dtype* data = inputs[0]->mutable_cpu_data();
  *data = static_cast<Dtype>(1);
}

template <typename Dtype>
int CustomSolver<Dtype>::GetActionFromNet() {
  int action = actionBlob_->cpu_data()[0];
  return action;
}

template <typename Dtype>
int CustomSolver<Dtype>::GetAction( float epsilon ) {
  CHECK_LE(0, epsilon);
  CHECK_LE(epsilon, 1);
  float r;
  caffe::caffe_rng_uniform(1, 0.0f, 1.0f, &r);
  if ( r < epsilon ) {
    return rand() % NUMBER_OF_LEGAL_ACTIONS;
  } else {
    return GetActionFromNet();
  }
}

template <typename Dtype>
void CustomSolver<Dtype>::FeedReward( int action, float reward ) {
  const Dtype* pred = predBlob_->cpu_data();
  for ( int i = 0; i < NUMBER_OF_LEGAL_ACTIONS; ++i )
    CHECK_EQ( pred[i], pred[i] );
  rewardBlob_->CopyFrom( *predBlob_, false, true );
  Dtype* rewardData = rewardBlob_->mutable_cpu_data();
  rewardData[rewardBlob_->offset(0, action, 0, 0)] = static_cast<Dtype>(reward);
}

template <typename Dtype>
shared_ptr<State<Dtype> > CustomSolver<Dtype>::PlayStep( shared_ptr<State<Dtype> > nowState, float & totalReward ) {
  int action;
  if ( this->iter_ <= REPLAY_START_SIZE )
    action = GetActionFromRandom();
  else {
    nowState->Feed( stateBlob_ );
    this->net_->ForwardTo( lossLayerID_ - 1 );
    action = GetAction( GetEpsilon() );
  }
  float reward;
  shared_ptr<State<Dtype> > state = environment_.Observe( action, reward );
  // LOG(INFO) << "PlayStep : observed (action, reward) = " << action << ", " << reward;
  history_.AddExperience( Transition<Dtype>(nowState, action, reward, state ) );
  totalReward += reward;
  return state;
}

template <typename Dtype>
float CustomSolver<Dtype>::TrainStep() {
  Transition<Dtype> trans = history_.Sample();
  float reward = trans.reward;
  const float GAMMA = 0.99;
  if ( trans.state_1 ) {
    trans.state_1->Feed( stateBlob_ );
    this->net_->ForwardTo( lossLayerID_ - 1 );
    int action = GetActionFromNet();
    //float pred = this->net_->blobs()[PREDICATE_BLOB_ID]->cpu_data()[action];
    float pred = this->net_->output_blobs()[0]->cpu_data()[1];
    CHECK_EQ( pred, pred );
    //LOG(INFO) << "TrainStep : pred = ", pred;
    reward += GAMMA * pred;
  }
  trans.state_0->Feed( stateBlob_ );
  FeedReward( trans.action, reward );
  float loss;
  this->net_->ForwardPrefilled( &loss );
  this->net_->Backward();
  return loss;
}

template <typename Dtype>
void CustomSolver<Dtype>::Step ( int iters ) {
  vector<Blob<Dtype>*> bottom_vec;
  const int start_iter = this->iter_;
  const int stop_iter = this->iter_ + iters;
  int average_loss = this->param_.average_loss();
  vector<Dtype> losses;
  Dtype smoothed_loss = 0;
  shared_ptr<State<Dtype> > nowState( (State<Dtype>*)NULL );
  float episodeReward = 0.0, totalReward = 0.0;
  int episodeCount = 0;

  while (this->iter_ < stop_iter) {
    // zero-init the params
    for (int i = 0; i < this->net_->params().size(); ++i) {
      shared_ptr<Blob<Dtype> > blob = this->net_->params()[i];
      switch (Caffe::mode()) {
      case Caffe::CPU:
        caffe_set(blob->count(), static_cast<Dtype>(0),
            blob->mutable_cpu_diff());
        break;
      case Caffe::GPU:
#ifndef CPU_ONLY
        caffe_gpu_set(blob->count(), static_cast<Dtype>(0),
            blob->mutable_gpu_diff());
#else
        NO_GPU;
#endif
        break;
      }
    }

    const bool display = this->param_.display() 
      && this->iter_ % this->param_.display() == 0;
    this->net_->set_debug_info(display && this->param_.debug_info());
    // accumulate the loss and gradient
    
    // This happens when game-over occurs, or at the first iteration.
    if ( !nowState ) {
      totalReward += episodeReward;
      if ( episodeCount ) {
        LOG(INFO) << "Episode #" << episodeCount << " ends with total score = "
          << episodeReward << ", average score = " << totalReward / episodeCount;
      }
      ++episodeCount;
      episodeReward = 0;
      environment_.ResetGame();
      nowState = environment_.GetState( true );
    }
    nowState = PlayStep( nowState, episodeReward );
    
    Dtype loss = 0;
    if ( this->iter_ > REPLAY_START_SIZE )
      for (int i = 0; i < this->param_.iter_size(); ++i) {
        loss += TrainStep();
      }
    loss /= this->param_.iter_size();
    // average the loss across iterations for smoothed reporting
    if (losses.size() < average_loss) {
      losses.push_back(loss);
      int size = losses.size();
      smoothed_loss = (smoothed_loss * (size - 1) + loss) / size;
    } else {
      int idx = (this->iter_ - start_iter) % average_loss;
      smoothed_loss += (loss - losses[idx]) / average_loss;
      losses[idx] = loss;
    }
    if (display) {
      LOG(INFO) << "Iteration " << this->iter_ << ", loss = " << smoothed_loss;
    }
    this->ApplyUpdate();

    // Increment the internal iter_ counter -- its value should always indicate
    // the number of times the weights have been updated.
    ++this->iter_;

    // Save a snapshot if needed.
    if (this->param_.snapshot() 
        && this->iter_ % this->param_.snapshot() == 0) {
      this->Snapshot();
    }
  }
}

#endif // __CUSTOM_SOLVER_H__
