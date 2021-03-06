CONFIG_FILE := Makefile.config
# Explicitly check for the config file, otherwise make -k will proceed anyway.
ifeq ($(wildcard $(CONFIG_FILE)),)
$(error $(CONFIG_FILE) not found. See $(CONFIG_FILE).example.)
endif
include $(CONFIG_FILE)

# generate train flags as specified in config file

train_flags += --solver=$(PROTO_DIR)/$(train_solver)
train_flags += --rom=$(ROM_DIR)/$(rom)
test_flags += --solver=$(PROTO_DIR)/$(test_solver)
test_flags += --rom=$(ROM_DIR)/$(rom)
test_flags += --snapshot=$(SNAPSHOT_DIR)/$(test_snapshot)

# generate LIBRARIES, INCLUDE_DIRS, LIBRARY_DIRS based on library directories given by config file

CAFFE_INCLUDE := $(CAFFE_ROOT)/include
CAFFE_LIB := $(CAFFE_ROOT)/build/lib
LIBRARIES += caffe
INCLUDE_DIRS += $(CAFFE_INCLUDE)
INCLUDE_DIRS += $(CAFFE_ROOT)/build/src
LIBRARY_DIRS += $(CAFFE_LIB)

LIBRARIES += cblas atlas	
INCLUDE_DIRS += $(BLAS_INCLUDE)
LIBRARY_DIRS += $(BLAS_LIB)

ALE_INCLUDE := $(ALE_ROOT)/src
ALE_LIB := $(ALE_ROOT)
LIBRARIES += ale
INCLUDE_DIRS += $(ALE_INCLUDE)
LIBRARY_DIRS += $(ALE_LIB)

LIBRARIES += SDL
INCLUDE_DIRS += $(SDL_INCLUDE)
CXXFLAGS += -D__USE_SDL

# complete generating flags

COMMON_FLAGS += -DCPU_ONLY
LIBRARIES += glog gflags protobuf leveldb snappy \
	lmdb boost_system hdf5_hl hdf5 m \
	opencv_core opencv_highgui opencv_imgproc
WARNINGS := -Wall -Wno-sign-compare

COMMON_FLAGS += $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))
CXXFLAGS += -pthread -fPIC $(COMMON_FLAGS)
LDFLAGS += $(foreach librarydir,$(LIBRARY_DIRS),-L$(librarydir)) $(PKG_CONFIG) \
		$(foreach library,$(LIBRARIES),-l$(library))
LINKFLAGS += -pthread -fPIC $(COMMON_FLAGS) $(WARNINGS)

# comment to disable quiet compilation
Q = @

EXEC := dqn
SRC_DIR := ./src
SRC_OBJECTS := $(patsubst %.cpp, %.o, $(wildcard $(SRC_DIR)/*.cpp))
HEADERS := $(wildcard $(SRC_DIR)/*.h)

all: $(EXEC)

$(EXEC): $(SRC_OBJECTS)
	$(Q) g++ $(SRC_OBJECTS) -o $@ $(LINKFLAGS) $(LDFLAGS) \
		-Wl,-rpath,$(CAFFE_LIB) 
	
%.o : %.cpp $(HEADERS)
	$(Q) g++ $< $(CXXFLAGS) -c -o $@
	
train: $(EXEC)
	./$(EXEC) train $(train_flags)
	
test: $(EXEC)
	./$(EXEC) test $(test_flags)

clean:
	rm $(EXEC)
	rm $(SRC_OBJECTS)

