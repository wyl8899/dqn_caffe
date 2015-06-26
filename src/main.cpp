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
class CustomSolver : public SGDSolver<Dtype> {
private:
  typedef SGDSolver<Dtype> super;
public:
  explicit CustomSolver( const SolverParameter& param )
    : super( param ) {}
  void Step( int );
  void GetInfo() {
  }
  void FeedState() {
    const vector<Blob<Dtype>*> & inputs = super::net_->input_blobs();
    Dtype* data = inputs[0]->mutable_cpu_data();
    *data = static_cast<Dtype>(1);
  }
  void FeedReward() {
    const vector<Blob<Dtype>*> & inputs = super::net_->input_blobs();
    Dtype* data = inputs[1]->mutable_cpu_data();
    // TODO : See doc
  }
};

template <typename Dtype>
void CustomSolver<Dtype>::Step ( int iters ) {
  vector<Blob<Dtype>*> bottom_vec;
  const int start_iter = super::iter_;
  const int stop_iter = super::iter_ + iters;
  int average_loss = this->param_.average_loss();
  vector<Dtype> losses;
  Dtype smoothed_loss = 0;

  while (super::iter_ < stop_iter) {
    // zero-init the params
    for (int i = 0; i < super::net_->params().size(); ++i) {
      shared_ptr<Blob<Dtype> > blob = super::net_->params()[i];
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
    
    // Feed
    FeedState();
    
    if (super::param_.test_interval() 
        && super::iter_ % super::param_.test_interval() == 0
        && (super::iter_ > 0 || super::param_.test_initialization())) {
      super::TestAll();
    }

    const bool display = super::param_.display() 
      && super::iter_ % super::param_.display() == 0;
    super::net_->set_debug_info(display && super::param_.debug_info());
    // accumulate the loss and gradient
    Dtype loss = 0;
    for (int i = 0; i < super::param_.iter_size(); ++i) {
      loss += super::net_->ForwardBackward(bottom_vec);
    }
    loss /= super::param_.iter_size();
    // average the loss across iterations for smoothed reporting
    if (losses.size() < average_loss) {
      losses.push_back(loss);
      int size = losses.size();
      smoothed_loss = (smoothed_loss * (size - 1) + loss) / size;
    } else {
      int idx = (super::iter_ - start_iter) % average_loss;
      smoothed_loss += (loss - losses[idx]) / average_loss;
      losses[idx] = loss;
    }
    if (display) {
      LOG(INFO) << "Iteration " << super::iter_ << ", loss = " << smoothed_loss;
      const vector<Blob<Dtype>*>& result = super::net_->output_blobs();
      int score_index = 0;
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        const string& output_name =
            super::net_->blob_names()[super::net_->output_blob_indices()[j]];
        const Dtype loss_weight =
            super::net_->blob_loss_weights()[super::net_->output_blob_indices()[j]];
        for (int k = 0; k < result[j]->count(); ++k) {
          ostringstream loss_msg_stream;
          if (loss_weight) {
            loss_msg_stream << " (* " << loss_weight
                            << " = " << loss_weight * result_vec[k] << " loss)";
          }
          LOG(INFO) << "    Train net output #"
              << score_index++ << ": " << output_name << " = "
              << result_vec[k] << loss_msg_stream.str();
        }
      }
      /*LOG(INFO) << "# of input: " << super::net_->num_inputs();
      const vector<Blob<Dtype>*> & inputs = super::net_->input_blobs();
      for ( int i = 0; i < inputs.size(); ++i ) {
        const Blob<Dtype> * blob = inputs[i];
        const string & blob_name =
            super::net_->blob_names()[super::net_->input_blob_indices()[i]];
        LOG(INFO) << " Train net input #"
            << i << ": " << blob_name << " has size: " << blob->count()
            << ", value: " << *blob->cpu_data() << endl;
      }*/
    }
    super::ApplyUpdate();

    // Increment the internal iter_ counter -- its value should always indicate
    // the number of times the weights have been updated.
    ++super::iter_;

    // Save a snapshot if needed.
    if (super::param_.snapshot() 
        && super::iter_ % super::param_.snapshot() == 0) {
      super::Snapshot();
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

  shared_ptr<caffe::Solver<float> >
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
