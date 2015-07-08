#include "CommonIncludes.h"
#include "DqnSolver.h"

typedef DqnSolver<float> DQN;

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
DEFINE_string(rom, "", 
    "The rom file of the game."); 
DEFINE_int32(display, 0,
    "Whether ALE Viz will be displayed.");

DEFINE_double(gamma, 0.99,
    "Discount factor used by Q value update.");
DEFINE_double(epsilon, 0.05,
    "Exploration used for evaluation.");
DEFINE_int32(learn_start, 50000,
    "Number of iteration before learn starts.");
DEFINE_int32(history_size, 1000000,
    "Number of transitions stored in replay memory.");
DEFINE_int32(update_freq, 4,
    "Number of actions taken between successive SGD updates.");
DEFINE_int32(frame_skip, 4,
    "Number of frames skipped between action selections.");
DEFINE_int32(clip_reward, 1,
    "Whether reward will be clipped to 1, 0, or -1 according to its sign.");
DEFINE_int32(eval_episodes, 3,
    "Number of episodes played during evaluation.");
DEFINE_int32(eval_freq, 10000,
    "Number of iterations between evaluations; 0 to disable.");
DEFINE_int32(sync_freq, 10000,
    "Number of iterations between target_net sync.");
DEFINE_int32(normalize, 0,
    "Whether normalize the gradient, i.e. divide diff by batch size.");

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

shared_ptr<DQN> caffe_init( int argc, char** argv) {
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
  
  shared_ptr<DQN> solver( new DQN( solver_param ) );
    
  return solver;
}

#ifdef __USE_SDL
  #include <SDL.h>
#endif

ALEInterface* ale_init() {
  static ALEInterface ale;
  ale.setFloat( "repeat_action_probability", 0.0 );
  if ( FLAGS_display ) {
    ale.setBool("display_screen", true);
    ale.setBool("sound", true);
  }
  ale.loadROM( FLAGS_rom );
  return &ale;
}

void Train( shared_ptr<DQN> solver ) {
  LOG(INFO) << "Starting Optimization";
  if ( FLAGS_snapshot.size() ) {
    LOG(FATAL) << "Restore from snapshot disabled "
      "due to the use of experience replay.";
  } else if ( FLAGS_weights.size() ) {
    CopyLayers( &*solver, FLAGS_weights );
    solver->Solve();
  } else {
    solver->Solve();
  }
  LOG(INFO) << "Optimization Done.";
}

void Test( shared_ptr<DQN> solver ) {
  string snapshot = FLAGS_snapshot;
  if ( snapshot.size() == 0 ) {
    LOG(FATAL) << "Needs a snapshot to evaluate.";
  } else {
    LOG(INFO) << "Restoring previous solver status from " << snapshot;
    solver->Restore( snapshot.c_str() );
    solver->Evaluate();
  }
}

int main( int argc, char** argv ) {
  string command( argv[1] );
  shared_ptr<DQN> solver = caffe_init( argc, argv );
  ALEInterface* ale = ale_init();
  solver->InitializeALE( ale );
  if ( command == "train" ) {
    Train( solver );
  } else if ( command == "test" ) {
    Test( solver );
  } else {
    LOG(ERROR) << "usage: dqn <command> <args>\n\n"
      "commands:\n"
      "  train           train or finetune a model\n"
      "  test            score a model\n";
    LOG(FATAL) << "Unknown command : " << command;
  }
  return 0;
}
