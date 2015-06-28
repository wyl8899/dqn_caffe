#include <ale_interface.hpp>

#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::shared_ptr;
using caffe::Timer;
using caffe::vector;

using caffe::SGDSolver;
using caffe::SolverParameter;
using caffe::caffe_set;

template <typename Dtype>
struct Scene {
  Dtype* data;
  int size;
  Scene( Dtype* d, int n ) {
    CHECK_GT( n, 0 );
    data = new Dtype[n];
    caffe::caffe_copy( n, d, data );
    size = n;
  }
  ~Scene() {
    delete [] data;
  }
  void Feed( Blob<Dtype>* blob ) {
    CHECK_EQ( size, blob->count() );
    Dtype* blobData = blob->mutable_cpu_data();
    caffe::caffe_copy( size, data, blobData );
  }
};

template <typename Dtype>
struct Transition {
  typedef Scene<Dtype> State;
  State scene_0, scene_1;
  int action;
  float reward;
  
  Experience( State s0, int a, float r, State s1 )
    : scene_0( s0 ), action( a ), reward( r ), scene_1( s1 ) {
  }
  
  void FeedScene( int i, Blob<Dtype>* blob ) {
    if( i == 0 )
      scene_0.Feed( blob );
    else if ( i == 1 )
      scene_1.Feed( blob );
    else
      LOG(FATAL) << "No such scene";
  }
};

template <typename Dtype>
class ExpHistory {
public:
  typedef Transition<Dtype> Exp;
  void Add( Exp & 
private:
  vector<Exp> history_;
  int currentIndex_;
};

template <typename Dtype>
class CustomSolver : public SGDSolver<Dtype> {
private:
  typedef SGDSolver<Dtype> super;
public:
  explicit CustomSolver( const SolverParameter& param )
    : super( param ) {
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
  void FeedState() {
    const vector<Blob<Dtype>*> & inputs = this->net_->input_blobs();
    Dtype* data = inputs[0]->mutable_cpu_data();
    *data = static_cast<Dtype>(1);
  }
  int GetAction() {
    if ( EpsilonGreedy( 0.1 ) ) {
      return rand() % 3;
    } else {
      Blob<Dtype>* actionBlob = this->net_->output_blobs()[0];
      int action = actionBlob->cpu_data()[0];
      return action;
    }
  }
  float ObserveReward( int action ) {
    static const int N = 3;
    static const float R[N] = { 1.0, 2.3, 1.6 };
    float reward;
    caffe::caffe_rng_gaussian(1, R[action], 0.5f, &reward);
    //reward = R[action];
    //LOG(INFO) << "Observed (action, reward) = " << action << ", " << reward; 
    return reward;
  }
  bool EpsilonGreedy( float epsilon ) {
    CHECK_LE(0, epsilon);
    CHECK_LE(epsilon, 1);
    float r;
    caffe::caffe_rng_uniform(1, 0.0f, 1.0f, &r);
    return r < epsilon;
  }
  void FeedReward( int action, float reward ) {
    const shared_ptr<Blob<Dtype> > rewardBlob = this->net_->blob_by_name("reward");
    const shared_ptr<Blob<Dtype> > predBlob = this->net_->blob_by_name("pred");
    const Dtype* pred = predBlob->cpu_data();
    //LOG(INFO) << "pred = " << pred[0] << ", " << pred[1] << ", " << pred[2];
    rewardBlob->CopyFrom(*predBlob, false, true);
    Dtype* rewardData = rewardBlob->mutable_cpu_data();
    rewardData[rewardBlob->offset(0, action, 0, 0)] = static_cast<Dtype>(reward);
  }
private:
  
};

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
    
    static int actionCount[3] = {0, 0, 0};
    
    Dtype loss = 0;
    for (int i = 0; i < this->param_.iter_size(); ++i) {
      const int lossLayerID = 3;
      FeedState();
      loss += this->net_->ForwardTo( lossLayerID - 1 );
      int action = GetAction();
      actionCount[action]++;
      FeedReward( action, ObserveReward( action ) );
      loss += this->net_->ForwardFrom( lossLayerID );
      this->net_->Backward();
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
      float perc[3], sC = 0;
      for ( int i = 0; i < 3; ++i )
        sC += actionCount[i];
      for ( int i = 0; i < 3; ++i )
        perc[i] = actionCount[i] / sC * 100;
      LOG(INFO) << "pred = " << pred[0] << ", " << pred[1] << ", " << pred[2];
      LOG(INFO) << "Percentage : " << perc[0] << ", " << perc[1] << ", " << perc[2];
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

// =====================================================================

DEFINE_int32(gpu, -1,
    "Run in GPU mode on given device ID.");
DEFINE_string(solver, "",
    "The solver definition protocol buffer text file.");
DEFINE_string(model, "",
    "The model definition protocol buffer text file..");
DEFINE_string(snapshot, "",
    "Optional; the snapshot solver state to resume training.");
DEFINE_string(weights, "",
    "Optional; the pretrained weights to initialize finetuning. "
    "Cannot be set simultaneously with snapshot.");
DEFINE_int32(iterations, 50,
    "The number of iterations to run.");

// Load the weights from the specified caffemodel(s) into the train and
// test nets.
void CopyLayers(caffe::Solver<float>* solver, const std::string& model_list) {
  std::vector<std::string> model_names;
  boost::split(model_names, model_list, boost::is_any_of(",") );
  for (int i = 0; i < model_names.size(); ++i) {
    LOG(INFO) << "Finetuning from " << model_names[i];
    solver->net()->CopyTrainedLayersFrom(model_names[i]);
    for (int j = 0; j < solver->test_nets().size(); ++j) {
      solver->test_nets()[j]->CopyTrainedLayersFrom(model_names[i]);
    }
  }
}

// =========================================================================

int main(int argc, char** argv) {
  // Print output to stderr (while still logging).
  FLAGS_alsologtostderr = 1;
  caffe::GlobalInit(&argc, &argv);
  CHECK_GT(FLAGS_solver.size(), 0) << "Need a solver definition to train.";
  CHECK(!FLAGS_snapshot.size() || !FLAGS_weights.size())
      << "Give a snapshot to resume training or weights to finetune "
      "but not both.";

  caffe::SolverParameter solver_param;
  caffe::ReadProtoFromTextFileOrDie(FLAGS_solver, &solver_param);

  // If the gpu flag is not provided, allow the mode and device to be set
  // in the solver prototxt.
  if (FLAGS_gpu < 0
      && solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU) {
    FLAGS_gpu = solver_param.device_id();
  }

  // Set device id and mode
  if (FLAGS_gpu >= 0) {
    LOG(INFO) << "Use GPU with device ID " << FLAGS_gpu;
    Caffe::SetDevice(FLAGS_gpu);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }

  LOG(INFO) << "Starting Optimization";

  // Use our modified solver
  shared_ptr<CustomSolver<float> >
    solver(new CustomSolver<float>(solver_param));

  if (FLAGS_snapshot.size()) {
    LOG(INFO) << "Resuming from " << FLAGS_snapshot;
    solver->Solve(FLAGS_snapshot);
  } else if (FLAGS_weights.size()) {
    CopyLayers(&*solver, FLAGS_weights);
    solver->Solve();
  } else {
    solver->Solve();
  }
  LOG(INFO) << "Optimization Done.";
  return 0;
}

// ==================================================================

#ifdef __USE_SDL
  #include <SDL.h>
#endif

int ale_main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " rom_file" << std::endl;
        return 1;
    }

    ALEInterface ale;

    // Get & Set the desired settings
    ale.setInt("random_seed", 123);
    //The default is already 0.25, this is just an example
    ale.setFloat("repeat_action_probability", 0.25);

#ifdef __USE_SDL
    ale.setBool("display_screen", true);
    ale.setBool("sound", true);
#endif

    // Load the ROM file. (Also resets the system for new settings to
    // take effect.)
    ale.loadROM(argv[1]);

    // Get the vector of legal actions
    ActionVect legal_actions = ale.getLegalActionSet();

    // Play 10 episodes
    for (int episode=0; episode<10; episode++) {
        float totalReward = 0;
        while (!ale.game_over()) {
            Action a = legal_actions[rand() % legal_actions.size()];
            // Apply the action and get the resulting reward
            float reward = ale.act(a);
            totalReward += reward;
        }
        cout << "Episode " << episode << " ended with score: " << totalReward << endl;
        ale.reset_game();
    }

    return 0;
}
