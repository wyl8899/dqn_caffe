#include "DqnSolver.h"

#include <cmath> // for sqrt

using caffe::BlobProto;

#define GET_BLOB(dest,net,name) \
  dest = NULL; \
  dest = net->blob_by_name(#name).get(); \
  CHECK( dest )

#define FIND_NET_BLOB(name) \
  GET_BLOB(name##Blob_,this->net_,name);

#define FIND_TARGET_NET_BLOB(name) \
  GET_BLOB(name##TargetBlob_,this->targetNet_,name);

template <typename Dtype>
DqnSolver<Dtype>::DqnSolver( const SolverParameter& param )
  : SGDSolver<Dtype>( param ), expHistory_( FLAGS_history_size ) {
  InitTargetNet();
  
  // cache information
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
  FIND_NET_BLOB(state);
  FIND_NET_BLOB(reward);
  FIND_NET_BLOB(pred);
  FIND_NET_BLOB(action);
  FIND_TARGET_NET_BLOB(state);
  FIND_TARGET_NET_BLOB(action);
  
  // build blobs for update value calculation
  const vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  sqGrad_.clear();
  tmpGrad_.clear();
  for (int i = 0; i < net_params.size(); ++i) {
    const vector<int>& shape = net_params[i]->shape();
    sqGrad_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    tmpGrad_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
  }
  
  // set hyperparameters
  gamma_ = FLAGS_gamma;
  epsilon_ = FLAGS_epsilon;
  learnStart_ = FLAGS_learn_start;
  updateFreq_ = FLAGS_update_freq;
  frameSkip_ = FLAGS_frame_skip;
  evalFreq_ = FLAGS_eval_freq;
  syncFreq_ = FLAGS_sync_freq;
}

template <typename Dtype>
void DqnSolver<Dtype>::InitTargetNet() {
  NetParameter netParam;
  CHECK( this->param_.has_net() );
  caffe::ReadNetParamsFromTextFileOrDie( this->param_.net(), &netParam );
  targetNet_.reset( new Net<Dtype>( netParam ) );
}

template <typename Dtype>
void DqnSolver<Dtype>::SyncTargetNet() {
  typedef shared_ptr<Layer<Dtype> > spLayer;
  typedef shared_ptr<Blob<Dtype> > spBlob;
  const vector<spLayer>& targetLayers = targetNet_->layers();
  const vector<spLayer>& sourceLayers = this->net_->layers();
  const int count = sourceLayers.size();
  for ( int i = 0; i < count; ++i ) {
    vector<spBlob>& sourceBlobs = sourceLayers[i]->blobs();
    vector<spBlob>& targetBlobs = targetLayers[i]->blobs();
    for ( int j = 0; j < sourceBlobs.size(); ++j ) {
      const bool reshape = false;
      targetBlobs[j]->CopyFrom( *sourceBlobs[j], false, reshape );
    }
  }
}

template <typename Dtype>
int DqnSolver<Dtype>::GetActionFromNet() {
  int action = actionBlob_->cpu_data()[0];
  return action;
}

template <typename Dtype>
float DqnSolver<Dtype>::GetEpsilon() {
  const float INITIAL_EPSILON = 1.0;
  const float FINAL_EPSILON = 0.1;
  const int FINAL_FRAME = 1000000; 
  float epsilon;
  if ( this->iter_ > FINAL_FRAME )
    epsilon = FINAL_EPSILON;
  else
    epsilon = INITIAL_EPSILON - 
      (INITIAL_EPSILON - FINAL_EPSILON) * ((float)this->iter_ / FINAL_FRAME);
  return epsilon;
}

template <typename Dtype>
int DqnSolver<Dtype>::GetAction( float epsilon ) {
  static int count[18] = {0};
  static const int display = 1000;
  static int count_total = 0;
  
  float r;
  caffe::caffe_rng_uniform(1, 0.0f, 1.0f, &r);
  if ( r < epsilon ) {
    return GetRandomAction();
  } else {
    int action = GetActionFromNet();
    count[action]++;
    count_total++;
    if ( count_total == display ) {
      count_total = 0;
      ostringstream ss;
      for ( int i = 0; i < legalActionCount_; ++i ) {
        ss << "[" << i << ": " << count[i] << "] ";
        count[i] = 0;
      }
      LOG(INFO) << ss.str();
    }
    return action;
  }
}

template <typename Dtype>
void DqnSolver<Dtype>::FeedReward( int action, float reward ) {
  const Dtype* pred = predBlob_->cpu_data();
  const bool reshape = true;
  rewardBlob_->CopyFrom( *predBlob_, false, reshape );
  Dtype* rewardData = rewardBlob_->mutable_cpu_data();
  Dtype actionPred = pred[action];
  float delta = reward - actionPred;
  // clip error
  if ( delta > 1.0 )
    delta = 1.0;
  if ( delta < -1.0 )
    delta = -1.0;
  rewardData[action] = static_cast<Dtype>(delta + actionPred);
}

template <typename Dtype>
State<Dtype> DqnSolver<Dtype>::PlayStep( State<Dtype> nowState, float* totalReward, float epsilon ) {
  int action;
  if ( this->iter_ <= learnStart_ )
    action = GetRandomAction();
  else {
    nowState.Feed( stateBlob_ );
    this->net_->ForwardTo( lossLayerID_ - 1 );
    action = GetAction( epsilon );
  }
  float reward;
  State<Dtype> state = environment_.Observe( action, &reward, frameSkip_ );
  expHistory_.AddExperience( Transition<Dtype>(nowState, action, reward, state ) );
  if ( totalReward )
    *totalReward += reward;
  return state;
}

template <typename Dtype>
Dtype DqnSolver<Dtype>::TrainStep() {
  Transition<Dtype> trans = expHistory_.Sample();
  float reward;
  if ( trans.state_1.isValid() ) {
    trans.state_1.Feed( stateTargetBlob_ );
    this->net_->ForwardTo( lossLayerID_ - 1 );
    int action = GetActionFromNet();
    float pred = actionTargetBlob_->cpu_data()[1];
    reward = trans.reward + gamma_ * pred;
  } else {
    reward = trans.reward;
  }
  trans.state_0.Feed( stateBlob_ );
  FeedReward( trans.action, reward );
  Dtype loss;
  this->net_->ForwardPrefilled( &loss );
  this->net_->Backward();
  return loss;
}

template <typename Dtype>
void DqnSolver<Dtype>::ApplyUpdate() {
  Dtype rate = this->GetLearningRate();
  this->ClipGradients();
  for (int param_id = 0; param_id < this->net_->params().size(); ++param_id) {
    this->Normalize(param_id);
    this->Regularize(param_id);
    ComputeUpdateValue(param_id, rate);
  }
  this->net_->Update();
}

template <typename Dtype>
void DqnSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
  const vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  const vector<float>& net_params_lr = this->net_->params_lr();
  const Dtype momentum = this->param_.momentum();
  const Dtype momentumC = 1.0 - momentum;
  Dtype local_rate = rate * net_params_lr[param_id];
  const int count = net_params[param_id]->count();
  shared_ptr<Blob<Dtype> > dw = net_params[param_id];
  shared_ptr<Blob<Dtype> > g = this->history_[param_id];
  shared_ptr<Blob<Dtype> > g2 = this->sqGrad_[param_id];
  shared_ptr<Blob<Dtype> > tmp = this->tmpGrad_[param_id];
  switch (Caffe::mode()) {
  case Caffe::CPU: {
    caffe::caffe_cpu_axpby( count, momentumC, dw->cpu_diff(),
      momentum, g->mutable_cpu_data() );
    caffe::caffe_mul( count, dw->cpu_diff(), dw->cpu_diff(),
      tmp->mutable_cpu_data() );
    caffe::caffe_cpu_axpby( count, momentumC, tmp->cpu_data(),
      momentum, g2->mutable_cpu_data() );
    for ( int i = 0; i < count; ++i ) {
      Dtype t = g2->cpu_data()[i];
      Dtype & diff = dw->mutable_cpu_diff()[i];
      t = sqrt( t ) + Dtype(0.01);
      diff = diff * local_rate / t;
    }
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
void DqnSolver<Dtype>::ZeroGradients() {
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
}

template <typename Dtype>
void DqnSolver<Dtype>::Step ( int iters ) {
  vector<Blob<Dtype>*> bottom_vec;
  const int start_iter = this->iter_;
  const int stop_iter = this->iter_ + iters;
  int average_loss = this->param_.average_loss();
  vector<Dtype> losses;
  Dtype smoothed_loss = 0;
  State<Dtype> nowState;

  while (this->iter_ < stop_iter) {
    ZeroGradients();
    const bool display = this->param_.display() 
      && this->iter_ % this->param_.display() == 0;
    const bool update = this->iter_ > learnStart_
      && this->iter_ % updateFreq_ == 0;
    const bool eval = evalFreq_ && this->iter_ % evalFreq_ == 0;
    const bool sync = this->iter_ % syncFreq_ == 0;
    this->net_->set_debug_info(display && this->param_.debug_info());
    
    // This happens when game-over occurs, or at the first iteration.
    if ( !nowState.isValid() ) {
      environment_.ResetGame();
      nowState = environment_.GetState( true );
    }
    nowState = PlayStep( nowState, NULL, GetEpsilon() );
    
    if ( update ) {
      for (int i = 0; i < this->param_.iter_size(); ++i) {
        TrainStep();
      }
      this->ApplyUpdate();
    }
    if ( sync ) {
      SyncTargetNet();
    }
    if ( eval ) {
      Evaluate();
    }
    if ( display ) {
      float epsilon = (this->iter_ > learnStart_) ? GetEpsilon() : 1.0;
      LOG(INFO) << "Iteration " << this->iter_ << ", epsilon = " << epsilon;
      LOG(INFO) << "  Average Q value = " << predBlob_->asum_data() / legalActionCount_;
    }

    ++this->iter_;

    if (this->param_.snapshot() 
        && this->iter_ % this->param_.snapshot() == 0) {
      this->Snapshot();
    }
  }
}

template <typename Dtype>
void DqnSolver<Dtype>::Solve( const char* resume_file ) {
  LOG(INFO) << "DqnSolver : Solving " << this->net_->name();
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
void DqnSolver<Dtype>::Evaluate() {
  const int count = FLAGS_eval_episodes;
  float reward = 0.0;
  State<Dtype> state;
  for ( int i = 0; i < count; ++i ) {
    environment_.ResetGame();
    state = environment_.GetState( true );
    while ( state.isValid() ) {
      state = PlayStep( state, &reward, epsilon_ );
    }
    // reward is accumulated to calculate average directly
  }
  LOG(INFO) << "Evaluate: Average score = " << reward / count
    << " over " << count << " game(s).";
}


template <typename Dtype>
void DqnSolver<Dtype>::SnapshotSolverState(SolverState* state) {
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
void DqnSolver<Dtype>::RestoreSolverState(const SolverState& state) {
  CHECK_EQ( state.history_size(), this->history_.size() + sqGrad_.size() )
      << "Incorrect length of history blobs.";
  LOG(INFO) << "DqnSolver: restoring history";
  for ( int i = 0; i < this->history_.size(); ++i ) {
    this->history_[i]->FromProto( state.history( i ) );
    sqGrad_[i]->FromProto( state.history( i + this->history_.size() ) );
  }
}

INSTANTIATE_CLASS(DqnSolver);
