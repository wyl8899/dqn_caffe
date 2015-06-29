#ifndef __CUSTOM_SOLVER_H__
#define __CUSTOM_SOLVER_H__

#include "CommonIncludes.h"
#include "State.h"
#include "Environment.h"
#include "Transition.h"
#include "ExpHistory.h"

template <typename Dtype>
class CustomSolver : public SGDSolver<Dtype> {
private:
  typedef SGDSolver<Dtype> super;
  enum {
    REPLAY_START_SIZE = 5000,
    HISTORY_SIZE = 100000,
    LOSS_LAYER_ID = 3
  };
public:
  explicit CustomSolver( const SolverParameter& param )
    : super( param ), history_( HISTORY_SIZE ) {
    const vector<string> & layers = this->net_->layer_names();
    for ( int i = 0; i < layers.size(); ++i ) {
      LOG(INFO) << "Layer #" << i << ": " << layers[i];
    }
    const vector<string> & blobs = this->net_->blob_names();
    for ( int i = 0; i < blobs.size(); ++i ) {
      LOG(INFO) << "Blob #" << i << ": " << blobs[i];
    }
  }
  void Step( int );
  
  void SetALE( ALEInterface* ale ) {
    environment_.SetALE( ale );
  }
  
private:
  ExpHistory<Dtype> history_;
  Environment<Dtype> environment_;
  
  void FeedState();
  int GetActionFromNet();
  int GetAction( float epsilon );
  void FeedReward( int action, float reward );
  
  void PlayStep( shared_ptr<State<Dtype> > state );
  float TrainStep();
};

template <typename Dtype>
void CustomSolver<Dtype>::FeedState() {
  const vector<Blob<Dtype>*> & inputs = this->net_->input_blobs();
  Dtype* data = inputs[0]->mutable_cpu_data();
  *data = static_cast<Dtype>(1);
}

template <typename Dtype>
int CustomSolver<Dtype>::GetActionFromNet() {
  Blob<Dtype>* actionBlob = this->net_->output_blobs()[0];
  int action = actionBlob->cpu_data()[0];
  return action;
}

template <typename Dtype>
int CustomSolver<Dtype>::GetAction( float epsilon ) {
  CHECK_LE(0, epsilon);
  CHECK_LE(epsilon, 1);
  float r;
  caffe::caffe_rng_uniform(1, 0.0f, 1.0f, &r);
  if ( r < epsilon || this->iter_ <= REPLAY_START_SIZE ) {
    return rand() % 3;
  } else {
    return GetActionFromNet();
  }
}

template <typename Dtype>
void CustomSolver<Dtype>::FeedReward( int action, float reward ) {
  const shared_ptr<Blob<Dtype> > rewardBlob = this->net_->blob_by_name("reward");
  const shared_ptr<Blob<Dtype> > predBlob = this->net_->blob_by_name("pred");
  const Dtype* pred = predBlob->cpu_data();
  //LOG(INFO) << "pred = " << pred[0] << ", " << pred[1] << ", " << pred[2];
  rewardBlob->CopyFrom(*predBlob, false, true);
  Dtype* rewardData = rewardBlob->mutable_cpu_data();
  rewardData[rewardBlob->offset(0, action, 0, 0)] = static_cast<Dtype>(reward);
}

template <typename Dtype>
void CustomSolver<Dtype>::PlayStep( shared_ptr<State<Dtype> > state ) {
  static int actionCount[3] = {0, 0, 0};
  state->Feed( this->net_->input_blobs()[0] );
  this->net_->ForwardTo( LOSS_LAYER_ID - 1 );
  int action = GetAction( 0.1 ); // Epsilon-Greedy with epsilon = 0.1
  float reward;
  shared_ptr<State<Dtype> > state_1 = environment_.Observe( action, reward );
  //LOG(INFO) << "Observed (action, reward) = " << action << ", " << reward;
  history_.AddExperience( Transition<Dtype>(state, action, reward, state_1 ) );
  actionCount[action]++;
}

template <typename Dtype>
float CustomSolver<Dtype>::TrainStep() {
  const int LOSS_LAYER_ID = 3;
  Transition<Dtype> trans = history_.Sample();
  trans.FeedState( 0, this->net_->input_blobs()[0] );
  this->net_->ForwardTo( LOSS_LAYER_ID - 1 );
  FeedReward( trans.action, trans.reward );
  float loss = this->net_->ForwardFrom( LOSS_LAYER_ID );
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
    
    static float dummy[1] = { 1.0 };
    PlayStep( shared_ptr<State<Dtype> >( new State<Dtype>( dummy, 1 ) ) );
    
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
      const shared_ptr<Blob<Dtype> > predBlob = this->net_->blob_by_name("pred");
      const Dtype* pred = predBlob->cpu_data();
      LOG(INFO) << "pred = " << pred[0] << ", " << pred[1] << ", " << pred[2];
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
