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
  SyncTargetNet();
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
float DqnSolver<Dtype>::GetEpsilon() {
  const float INITIAL_EPSILON = 1.0;
  const float FINAL_EPSILON = 0.1;
  const int FINAL_FRAME = 1000000; 
  float epsilon;
  if ( this->iter_ <= learnStart_)
    epsilon = 1.0;
  else if ( this->iter_ > FINAL_FRAME )
    epsilon = FINAL_EPSILON;
  else
    epsilon = INITIAL_EPSILON - 
      (INITIAL_EPSILON - FINAL_EPSILON) * ((float)this->iter_ / FINAL_FRAME);
  return epsilon;
}

template <typename Dtype>
int DqnSolver<Dtype>::GetAction( State<Dtype> state, float epsilon ) {
  float r;
  caffe::caffe_rng_uniform(1, 0.0f, 1.0f, &r);
  if ( r < epsilon ) {
    return GetRandomAction();
  } else {
    state.Feed( stateBlob_ );
    this->net_->ForwardTo( lossLayerID_ - 1 );
    int action = actionBlob_->cpu_data()[0];
    actionLog_.Add( action );
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
  int action = GetAction( nowState, epsilon );
  float reward;
  State<Dtype> state = environment_.Observe( action, &reward, frameSkip_ );
  expHistory_.AddExperience( Transition<Dtype>(nowState, action, reward, state ) );
  if ( totalReward )
    *totalReward += reward;
  return state;
}

template <typename Dtype>
void DqnSolver<Dtype>::TrainStep() {
  Transition<Dtype> trans = expHistory_.Sample();
  float reward;
  // for transition (s, a, r, s'):
  //   expected_reward = r, if s' is terminal
  //   expected_reward = r + gamma * max(a'){target_Q(s', a')}, otherwise
  if ( trans.state_1.isValid() ) {
    trans.state_1.Feed( stateTargetBlob_ );
    targetNet_->ForwardTo( lossLayerID_ - 1 );
    float pred = actionTargetBlob_->cpu_data()[1];
    reward = trans.reward + gamma_ * pred;
  } else {
    reward = trans.reward;
  }
  // accumulate the gradient of Loss = (expected_reward - Q(s, a))^2
  trans.state_0.Feed( stateBlob_ );
  this->net_->ForwardTo( lossLayerID_ - 1 );
  FeedReward( trans.action, reward );
  this->net_->ForwardFrom( lossLayerID_ );
  this->net_->Backward();
}

template <typename Dtype>
void DqnSolver<Dtype>::ApplyUpdate() {
  Dtype rate = this->GetLearningRate();
  this->ClipGradients();
  for ( int param_id = 0; param_id < this->net_->params().size(); ++param_id ) {
    if ( FLAGS_normalize )
      this->Normalize( param_id );
    this->Regularize( param_id );
    ComputeUpdateValue( param_id, rate );
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
    caffe::caffe_mul( count, g->cpu_data(), g->cpu_data(),
      tmp->mutable_cpu_data() );
    caffe::caffe_cpu_axpby( count, Dtype(1), g2->cpu_data(),
      Dtype(-1), tmp->mutable_cpu_data() );
    caffe::caffe_add_scalar( count, Dtype(0.01), tmp->mutable_cpu_data() );
    for ( int i = 0; i < count; ++i ) {
      Dtype t = tmp->cpu_data()[i];
      Dtype& diff = dw->mutable_cpu_diff()[i];
      diff = diff / sqrt( t );
    }
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
void DqnSolver<Dtype>::Solve( const char* resume_file ) {
  LOG(INFO) << "DqnSolver : Solving " << this->net_->name();
  LOG(INFO) << "Learning Rate Policy: " << this->param_.lr_policy();

  if (resume_file) {
    LOG(INFO) << "Restoring previous solver status from " << resume_file;
    this->Restore(resume_file);
  }
  
  const int stopIter = this->param_.max_iter();
  while ( this->iter_ < stopIter ) {
    actionLog_.Clear();
    for ( int i = 0; i < evalFreq_; ++i ) {
      bool learn = true;
      PlayEpisode( learn, GetEpsilon() );
    }
    actionLog_.Report();
    
    actionLog_.Clear();
    Evaluate();
    actionLog_.Report();
  }
 
  LOG(INFO) << "Optimization Done.";
}

template <typename Dtype>
float DqnSolver<Dtype>::PlayEpisode( bool learn, float epsilon ) {
  float reward = 0.0;
  environment_.ResetGame();
  State<Dtype> state = environment_.GetState( true );
  while ( state.isValid() ) {
    state = PlayStep( state, &reward, epsilon );
    const bool display = this->param_.display() 
      && this->iter_ % this->param_.display() == 0;
    if ( learn ) {
      if ( this->iter_ > learnStart_ ) {
        if ( this->iter_ % updateFreq_ == 0 ) {
          ZeroGradients();
          for (int i = 0; i < this->param_.iter_size(); ++i) {
            TrainStep();
          }
          this->ApplyUpdate();
        }
        if ( this->iter_ % syncFreq_ == 0 ) {
          SyncTargetNet();
        }
        if ( display ) {
          LOG(INFO) << "Iteration " << this->iter_ << ", epsilon = " << epsilon;
          LOG(INFO) << "  Average Q value = " << predBlob_->asum_data() / legalActionCount_;
        }
        if ( this->param_.snapshot() 
          && this->iter_ % this->param_.snapshot() == 0 ) {
          this->Snapshot();
        }
      }
      ++this->iter_;
    }
  }
  return reward;
}

template <typename Dtype>
void DqnSolver<Dtype>::Evaluate() {
  const int count = FLAGS_eval_episodes;
  float totalReward = 0.0;
  State<Dtype> state;
  for ( int i = 0; i < count; ++i ) {
    const bool update = false;
    float reward = PlayEpisode( update, epsilon_ );
    LOG(INFO) << "    Evaluate: Episode #" << i << " ends with score " << reward;
    totalReward += reward;
  }
  LOG(INFO) << "  Evaluate: Average score = " << totalReward / count
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
