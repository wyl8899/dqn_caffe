#include "CommonIncludes.h"
#include "CustomSolver.h"

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
DEFINE_string( rom, "", 
    "The rom file of the game." ); 

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

shared_ptr<CustomSolver<float> > caffe_init( int argc, char** argv) {
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
  
  shared_ptr<CustomSolver<float> >
    solver(new CustomSolver<float>(solver_param));
    
  return solver;
}

#ifdef __USE_SDL
  #include <SDL.h>
#endif

ALEInterface* ale_init() {
  static ALEInterface ale;
  ale.setFloat( "repeat_action_probability", 0.0 );
#ifdef __USE_SDL
  //ale.setBool("display_screen", true);
  //ale.setBool("sound", true);
#endif
  ale.loadROM( FLAGS_rom );
  return &ale;
}

int main( int argc, char** argv ) {
  shared_ptr<CustomSolver<float> > solver = caffe_init( argc, argv );
  ALEInterface* ale = ale_init();
  solver->SetALE( ale );
  
  LOG(INFO) << "Starting Optimization";
  if ( FLAGS_snapshot.size() ) {
    LOG(INFO) << "Resuming from " << FLAGS_snapshot;
    solver->Solve( FLAGS_snapshot );
  } else if ( FLAGS_weights.size() ) {
    CopyLayers( &*solver, FLAGS_weights );
    solver->Solve();
  } else {
    solver->Solve();
  }
  LOG(INFO) << "Optimization Done.";
  
  return 0;
}
