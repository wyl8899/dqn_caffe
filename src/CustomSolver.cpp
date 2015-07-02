#include "CustomSolver.h"

#include <cmath> // for sqrt

using caffe::BlobProto;

#define FIND_BLOB(name) \
  name##Blob_ = NULL; \
  name##Blob_ = this->net_->blob_by_name(#name).get(); \
  CHECK( name##Blob_ );

template <typename Dtype>
CustomSolver<Dtype>::CustomSolver( const SolverParameter& param )
  : SGDSolver<Dtype>( param ), expHistory_( HISTORY_SIZE ) {
  const vector<string> & layers = this->net_->layer_names();
  for ( int i = 0; i < layers.size(); ++i ) {
    LOG(INFO) << "Layer #" << i << ": " << layers[i];
    if ( layers[i] == "loss" )
      lossLayerID_ = i;
  }
  const vector<string> & blobs = this->net_->blob_names();
  for ( int i = 0; i < blobs.size(); ++i ) {
    LOG(INFO) << "Blob #" << i << ": " << blobs[i];
  }
  FIND_BLOB(state);
  FIND_BLOB(reward);
  FIND_BLOB(pred);
  FIND_BLOB(action);
  
  const vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  sqGrad_.clear();
  tmpGrad_.clear();
  for (int i = 0; i < net_params.size(); ++i) {
    const vector<int>& shape = net_params[i]->shape();
    sqGrad_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    tmpGrad_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
  }
}

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
int CustomSolver<Dtype>::GetAction() {
  const float INITIAL_EPSILON = 1.0;
  const float FINAL_EPSILON = 0.1;
  const int FINAL_FRAME = 1000000; 
  
  float epsilon;
  if ( this->iter_ > FINAL_FRAME )
    epsilon = FINAL_EPSILON;
  else
    epsilon = INITIAL_EPSILON - 
      (INITIAL_EPSILON - FINAL_EPSILON) * ((float)this->iter_ / FINAL_FRAME);
  float r;
  caffe::caffe_rng_uniform(1, 0.0f, 1.0f, &r);
  if ( r < epsilon ) {
    return GetRandomAction();
  } else {
    return GetActionFromNet();
  }
}

template <typename Dtype>
void CustomSolver<Dtype>::FeedReward( int action, float reward ) {
  const Dtype* pred = predBlob_->cpu_data();
  rewardBlob_->CopyFrom( *predBlob_, false, true );
  Dtype* rewardData = rewardBlob_->mutable_cpu_data();
  Dtype actionPred = pred[action];
  reward -= actionPred;
  if ( reward > 1.0 )
    reward = 1.0;
  if ( reward < -1.0 )
    reward = -1.0;
  rewardData[action] = static_cast<Dtype>(reward + actionPred);
}

template <typename Dtype>
State<Dtype> CustomSolver<Dtype>::PlayStep( State<Dtype> nowState, float & totalReward ) {
  int action;
  if ( this->iter_ <= REPLAY_START_SIZE )
    action = GetRandomAction();
  else {
    nowState.Feed( stateBlob_ );
    this->net_->ForwardTo( lossLayerID_ - 1 );
    action = GetAction();
  }
  float reward;
  State<Dtype> state = environment_.Observe( action, reward );
  //state.inspect( "PlayStep()" );
  // LOG(INFO) << "PlayStep : observed (action, reward) = " << action << ", " << reward;
  expHistory_.AddExperience( Transition<Dtype>(nowState, action, reward, state ) );
  totalReward += reward;
  return state;
}

template <typename Dtype>
Dtype CustomSolver<Dtype>::TrainStep() {
  Transition<Dtype> trans = expHistory_.Sample();
  float reward = trans.reward;
  const float GAMMA = 0.99;
  if ( trans.state_1.isValid() ) {
    trans.state_1.Feed( stateBlob_ );
    this->net_->ForwardTo( lossLayerID_ - 1 );
    int action = GetActionFromNet();
    float pred = actionBlob_->cpu_data()[1];
    // CHECK_EQ( pred, pred );
    reward += GAMMA * pred;
  }
  trans.state_0.Feed( stateBlob_ );
  FeedReward( trans.action, reward );
  Dtype loss;
  this->net_->ForwardPrefilled( &loss );
  this->net_->Backward();
  return loss;
}

template <typename Dtype>
void CustomSolver<Dtype>::ApplyUpdate() {
  Dtype rate = this->GetLearningRate();
  if (this->param_.display() && this->iter_ % this->param_.display() == 0) {
    LOG(INFO) << "Iteration " << this->iter_ << ", lr = " << rate;
  }
  this->ClipGradients();
  for (int param_id = 0; param_id < this->net_->params().size(); ++param_id) {
    this->Normalize(param_id);
    this->Regularize(param_id);
    ComputeUpdateValue(param_id, rate);
  }
  this->net_->Update();
}

template <typename Dtype>
void CustomSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
  const vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  const vector<float>& net_params_lr = this->net_->params_lr();
  Dtype momentum = this->param_.momentum();
  Dtype momentumC = 1.0 - momentum;
  Dtype local_rate = rate * net_params_lr[param_id];
  const int count = net_params[param_id]->count();
  shared_ptr<Blob<Dtype> > dw = net_params[param_id];
  shared_ptr<Blob<Dtype> > g = this->history_[param_id];
  shared_ptr<Blob<Dtype> > g2 = this->sqGrad_[param_id];
  shared_ptr<Blob<Dtype> > tmp = this->tmpGrad_[param_id];
  // Compute the update to history, then copy it to the parameter diff.
  switch (Caffe::mode()) {
  case Caffe::CPU: {
    caffe::caffe_cpu_axpby( count, momentumC, dw->cpu_diff(),
      momentum, g->mutable_cpu_data() );
    caffe::caffe_mul( count, dw->cpu_diff(), dw->cpu_diff(),
      tmp->mutable_cpu_data() );
    caffe::caffe_cpu_axpby( count, momentumC, tmp->cpu_data(),
      momentum, g2->mutable_cpu_data() );
    caffe::caffe_mul( count, g->cpu_data(), g->cpu_data(),
      tmp->mutable_cpu_data() );
    caffe::caffe_cpu_axpby( count, Dtype(1), g2->cpu_data(),
      Dtype(-1), tmp->mutable_cpu_data() );
    caffe::caffe_add_scalar( count, Dtype(0.01), tmp->mutable_cpu_data() );
    // TODO : use element-wise sqrt provided by library
    for ( int i = 0; i < count; ++i ) {
      Dtype & t = tmp->mutable_cpu_data()[i];
      t = sqrt( t );
    }
    caffe::caffe_div( count, dw->cpu_diff(), tmp->cpu_data(),
      dw->mutable_cpu_diff() );
    caffe::caffe_scal( count, local_rate, dw->mutable_cpu_diff() );
    break;
  }
  case Caffe::GPU: {
    NOT_IMPLEMENTED;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

template <typename Dtype>
void CustomSolver<Dtype>::Step ( int iters ) {
  vector<Blob<Dtype>*> bottom_vec;
  const int start_iter = this->iter_;
  const int stop_iter = this->iter_ + iters;
  int average_loss = this->param_.average_loss();
  vector<Dtype> losses;
  Dtype smoothed_loss = 0;
  State<Dtype> nowState;
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
    
    // Our agent selects UPDATE_FREQUENCY actions between successive SGD update.
    for ( int i = 0; i < UPDATE_FREQUENCY; ++i ){
      // This happens when game-over occurs, or at the first iteration.
      if ( !nowState.isValid() ) {
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
    }
    
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

template <typename Dtype>
void CustomSolver<Dtype>::Solve( const char* resume_file ) {
  LOG(INFO) << "CustomSolver : Solving " << this->net_->name();
  LOG(INFO) << "Learning Rate Policy: " << this->param_.lr_policy();

  if (resume_file) {
    LOG(INFO) << "Restoring previous solver status from " << resume_file;
    this->Restore(resume_file);
  }

  Step(this->param_.max_iter() - this->iter_);
  
  // If we haven't already, save a snapshot after optimization, unless
  // overridden by setting snapshot_after_train := false
  if (this->param_.snapshot_after_train()
      && (!this->param_.snapshot() || this->iter_ % this->param_.snapshot() != 0)) {
    this->Snapshot();
  }
  // After the optimization is done, run an additional train and test pass to
  // display the train and test loss/outputs if appropriate (based on the
  // display and test_interval settings, respectively).  Unlike in the rest of
  // training, for the train net we only run a forward pass as we've already
  // updated the parameters "max_iter" times -- this final pass is only done to
  // display the loss, which is computed in the forward pass.
  if (this->param_.display() && this->iter_ % this->param_.display() == 0) {
    Dtype loss;
    this->net_->ForwardPrefilled(&loss);
    LOG(INFO) << "Iteration " << this->iter_ << ", loss = " << loss;
  }
  if (this->param_.test_interval() && this->iter_ % this->param_.test_interval() == 0) {
    this->TestAll();
  }
  LOG(INFO) << "Optimization Done.";
}


template <typename Dtype>
void CustomSolver<Dtype>::SnapshotSolverState(SolverState* state) {
  state->clear_history();
  for ( int i = 0; i < this->history_.size(); ++i ) {
    // Add gradient history
    BlobProto* history_blob = state->add_history();
    this->history_[i]->ToProto( history_blob );
  }
  for ( int i = 0; i < sqGrad_.size(); ++i ) {
    // Add squared gradient history
    BlobProto* history_blob = state->add_history();
    sqGrad_[i]->ToProto( history_blob );
  }
}

template <typename Dtype>
void CustomSolver<Dtype>::RestoreSolverState(const SolverState& state) {
  CHECK_EQ( state.history_size(), this->history_.size() + sqGrad_.size() )
      << "Incorrect length of history blobs.";
  LOG(INFO) << "CustomSolver: restoring history";
  for ( int i = 0; i < this->history_.size(); ++i ) {
    this->history_[i]->FromProto( state.history( i ) );
    sqGrad_[i]->FromProto( state.history( i + this->history_.size() ) );
  }
}

INSTANTIATE_CLASS(CustomSolver);
