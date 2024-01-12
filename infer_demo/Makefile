TRT=/media/ros/A666B94D66B91F4D/ros/new_deploy/TensorRT-8.5.3.1.Linux.x86_64-gnu.cuda-11.8.cudnn8.6/TensorRT-8.5.3.1/
CUDA=/usr/local/cuda-11.6
INCLUDE=-I$(CUDA)/include/ \
-I$(TRT)/include \
-Isrc

CXX_FLAGS=-std=c++11 -O0 -g -Wno-deprecated-declarations
TRT_LIB = $(TRT)/lib/

LIBS=-L$(CUDA)/lib64/ -lcudart \
-lcublas \
-lcublasLt \
-lcurand \
-lcusolver \
-lcudnn \
-lcudnn_adv_infer \
-lcudnn_cnn_infer \
-lcudnn_ops_infer \
$(TRT_LIB)libnvinfer.so \
$(TRT_LIB)libnvparsers.so \
$(TRT_LIB)libnvonnxparser.so \
$(TRT_LIB)libnvcaffe_parser.so \
$(TRT_LIB)libnvinfer_plugin.so \
-lrt -ldl -lpthread -rdynamic

ernie_test:
	g++ src/ernie_infer_demo.cpp src/trt_helper.cc -o ernie_infer_demo \
		$(CXX_FLAGS) $(INCLUDE) -Xlinker "-(" $(LIBS) -Xlinker "-)"
ernie_test2:
	g++ src/ernie_infer_demo.cpp src/trt_helper.cc -o ernie_infer_demo_cuda_graph \
		$(CXX_FLAGS) $(INCLUDE) -Xlinker "-(" $(LIBS) -Xlinker "-)"