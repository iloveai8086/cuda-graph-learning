#include "trt_helper.h"

#include <string>
#include <fstream>
#include <sstream>

#include "NvInferPlugin.h"

using namespace std;

// BEGIN_LIB_NAMESPACE {

cuda_shared_ptr<void> CpuToDevice(const std::vector<int> &shape, int *data_ptr) {
    void *d_ptr;
    auto cpu_ptr = static_cast<void *>(data_ptr);
    int data_size = 1;
    for (int i = 0; i < shape.size(); i++) data_size *= shape[i];
    auto ret = cudaMalloc(&d_ptr, data_size * sizeof(int));
    //printf("int memory\n");
    if (ret) printf("memory error\n");
    ret = cudaMemcpy(d_ptr, cpu_ptr, data_size * sizeof(int), cudaMemcpyHostToDevice);
    if (ret) printf("memory error\n");
    cuda_shared_ptr<void> cuda_ptr;
    make_cuda_shared(cuda_ptr, d_ptr);
    return cuda_ptr;
}

cuda_shared_ptr<void> CpuToDevice(const std::vector<int> &shape, float *data_ptr) {
    void *d_ptr;
    auto cpu_ptr = static_cast<void *>(data_ptr);
    int data_size = 1;
    for (int i = 0; i < shape.size(); i++) data_size *= shape[i];
    auto ret = cudaMalloc(&d_ptr, data_size * sizeof(float));
    //printf("float memory\n");
    if (ret) printf("memory error\n");
    ret = cudaMemcpy(d_ptr, cpu_ptr, data_size * sizeof(float), cudaMemcpyHostToDevice);
    if (ret) printf("memory error\n");
    cuda_shared_ptr<void> cuda_ptr;
    make_cuda_shared(cuda_ptr, d_ptr);
    return cuda_ptr;
}

void DeviceToCpu(const std::vector<int> &shape, cuda_shared_ptr<void> cuda_ptr, float *data_ptr) {
    int data_size = 1;
    for (int i = 0; i < shape.size(); i++) data_size *= shape[i];
    if (data_size == 0) {
        std::cout << "data_size == 0" << std::endl;
        assert(0);
    }
    auto d_ptr = static_cast<void *>(data_ptr);
    auto ret = cudaMemcpy(d_ptr, cuda_ptr.get(), data_size * sizeof(float), cudaMemcpyDeviceToHost);
    printf("copy back\n");
    if (ret) printf("memory error\n");
}

TrtLogger::TrtLogger(nvinfer1::ILogger::Severity level) : level_(level) {}

nvinfer1::ILogger &TrtLogger::getTRTLogger() { return *this; }

// trt logger
void TrtLogger::log(Severity severity, const char *msg)

noexcept {
if (severity > level_) {
return;
}

switch (severity) {
    case Severity::kINTERNAL_ERROR:
        std::cout << "[TRT] " << std::string(msg) << std::endl;
        break;
    case Severity::kERROR:
        std::cout << "[TRT] " << std::string(msg) << std::endl;
        break;
    case Severity::kWARNING:
        std::cout << "[TRT] " <<std::string(msg) << std::endl;
        break;
    case Severity::kINFO:
        std::cout << "[TRT] " << std::string(msg) << std::endl;
        break;
    case Severity::kVERBOSE:
        std::cout << "[TRT] " << std::string(msg) << std::endl;
    }
}


TrtHepler::TrtHepler(std::string model_param, int dev_id)
        : _dev_id(dev_id), _model_param(model_param) {
    { // read model, deserializeCudaEngine and createExecutionContext
        std::ifstream t(_model_param);  // string pth
        std::stringstream buffer;
        buffer << t.rdbuf();
        std::string contents(buffer.str());

        CUDA_CHECK(cudaSetDevice(_dev_id));
        CUDA_CHECK(cudaStreamCreate(&cuda_stream_));

        TrtLogger trt_logger;
        initLibNvInferPlugins(&trt_logger.getTRTLogger(), "");
        auto runtime = MakeUnique(nvinfer1::createInferRuntime(trt_logger.getTRTLogger()));
        auto e = runtime->deserializeCudaEngine((void *) contents.c_str(),
                                                contents.size(), nullptr);
        engine_ = MakeShared(e);
        context_ = MakeShared(engine_->createExecutionContext());
        context_->setOptimizationProfile(0);
    }

}

int TrtHepler::Forward(sample &s) {
    cudaSetDevice(_dev_id);
    auto rc_ids_tensor = CpuToDevice(s.shape_info_0, s.i0.data());
    auto sent_ids_tensor = CpuToDevice(s.shape_info_1, s.i1.data());
    auto pos_ids_tensor = CpuToDevice(s.shape_info_2, s.i2.data());
    auto input_mask_tensor = CpuToDevice(s.shape_info_3, s.i3.data());
    auto tmp6_tensor = CpuToDevice(s.shape_info_4, s.i4.data());
    auto tmp7_tensor = CpuToDevice(s.shape_info_5, s.i5.data());
    auto tmp8_tensor = CpuToDevice(s.shape_info_6, s.i6.data());
    auto tmp9_tensor = CpuToDevice(s.shape_info_7, s.i7.data());
    auto tmp10_tensor = CpuToDevice(s.shape_info_8, s.i8.data());
    auto tmp11_tensor = CpuToDevice(s.shape_info_9, s.i9.data());
    auto tmp12_tensor = CpuToDevice(s.shape_info_10, s.i10.data());
    auto tmp13_tensor = CpuToDevice(s.shape_info_11, s.i11.data());

    void *out_ptr;
    auto ret_ = cudaMalloc(&out_ptr, s.shape_info_0[0] * sizeof(float));  // -1 * 1
    cuda_shared_ptr<void> cuda_out_ptr;
    make_cuda_shared(cuda_out_ptr, out_ptr);

    cudaEvent_t start, stop;
    float elapsed_time = 0.0;

    int binding_idx = 0;
    //std::vector<std::vector<int>> input_dims = {s.shape_info_0, s.shape_info_1, s.shape_info_2, s.shape_info_3,
    //s.shape_info_4, s.shape_info_5, s.shape_info_6, s.shape_info_7,
    //s.shape_info_8, s.shape_info_9, s.shape_info_10, s.shape_info_11};
    std::vector <std::vector<int>> input_dims = {s.shape_info_0, s.shape_info_1, s.shape_info_2, s.shape_info_3,
                                                 s.shape_info_4, s.shape_info_5, s.shape_info_6, s.shape_info_7,
                                                 s.shape_info_8, s.shape_info_9, s.shape_info_10, s.shape_info_11};
    auto num_bindings = engine_->getNbBindings();
    printf("num_bindings:%d\n", num_bindings);
    // set device_bindings_ and setBindingDimensions
    for (size_t i = 0; i < input_dims.size(); i++) {
        std::vector<int> dims_vec = input_dims[i];
        nvinfer1::Dims trt_dims;
        trt_dims.nbDims = static_cast<int>(dims_vec.size());
        memcpy(trt_dims.d, dims_vec.data(), sizeof(int) * trt_dims.nbDims);
        context_->setBindingDimensions(binding_idx, trt_dims);
        binding_idx++;
    }

    if (!context_->allInputDimensionsSpecified()) {
        //gLogFatal << "context_->allInputDimensionsSpecified() error";
        std::cout << ("context_->allInputDimensionsSpecified() error") << std::endl;
        assert(0);
    }

    // set the input dim

    void *device_bindings[13] = {rc_ids_tensor.get(), sent_ids_tensor.get(), pos_ids_tensor.get(),
                                 input_mask_tensor.get(),
                                 tmp6_tensor.get(), tmp7_tensor.get(),
                                 tmp8_tensor.get(), tmp9_tensor.get(), tmp10_tensor.get(),
                                 tmp11_tensor.get(), tmp12_tensor.get(), tmp13_tensor.get(),
                                 cuda_out_ptr.get()};
    // printf("before enqueue\n");
    bool ret = context_->enqueueV2(device_bindings, cuda_stream_, nullptr);
    if (!ret) {
        std::cout << ("context_->enqueueV2 failed!") << std::endl;
        return -100;
    }

    cudaMemcpy(s.out_data.data(), cuda_out_ptr.get(), s.shape_info_0[0] * sizeof(float), cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(cuda_stream_);
    struct timeval tv;
    gettimeofday(&tv, NULL);
    s.timestamp = tv.tv_sec * 1000000 + tv.tv_usec;

}

TrtHepler::~TrtHepler() {
    CUDA_CHECK(cudaStreamDestroy(cuda_stream_));
}

// ---------------------------------------------------------------------------------------------------------
// 新增的代码
TrtEngine::TrtEngine(std::string model_param, int dev_id) : dev_id_(dev_id), _model_param(model_param) {
    std::ifstream t(_model_param);
    std::stringstream buffer;
    buffer << t.rdbuf();
    std::string contents(buffer.str());

    CUDA_CHECK(cudaSetDevice(dev_id_));
    initLibNvInferPlugins(&trt_logger.getTRTLogger(), "");
    auto runtime = MakeUnique(nvinfer1::createInferRuntime(trt_logger.getTRTLogger()));
    auto e = runtime->deserializeCudaEngine((void *) contents.c_str(), contents.size(), nullptr);
    engine_ = MakeShared(e);
    std::cout << "getNbIOTensors: " << engine_->getNbBindings() << std::endl;
}

// 老师那边最大的profile是128的seq-len，我这边是64，感觉这个变量应该是要改的
constexpr size_t kAlignment = 64;
constexpr int ceildiv(int a, int b) {
    return (a + b - 1) / b;
}

constexpr int AlignTo(int a, int b = kAlignment) {
    return ceildiv(a, b) * b;
}

TrtContext::TrtContext(TrtEngine *trt_engine, int profile_idx) {
    profile_idx_ = profile_idx;
    engine_ = trt_engine->engine_;
    dev_id_ = trt_engine->dev_id_;
    CUDA_CHECK(cudaSetDevice(dev_id_));
    CUDA_CHECK(cudaStreamCreate(&cuda_stream_));
    context_ = MakeShared(engine_->createExecutionContext());
    context_->setOptimizationProfile(profile_idx_);

    start_binding_idx_ = profile_idx * engine_->getNbBindings() / engine_->getNbOptimizationProfiles();
    auto max_profile = engine_->getProfileDimensions(start_binding_idx_, profile_idx, nvinfer1::OptProfileSelector::kMAX);
    max_batch_ = max_profile.d[0];
    max_seq_len_ = max_profile.d[1];

    // 4 inputs [B, S, 1]
    // 这是把mask也当int了，应该是简化了mask了
    // 根据当前的profile，申请内存，此时的每一个profile里面的min、opt、max应该都是一样的
    align_input_bytes_ = AlignTo(max_batch_ * max_seq_len_ * sizeof(int));
    // 举个例子
    // 假如我的输入是(10,63) -> (630 + 64 - 1) / 64 -> 40
    // 40 * 64 -> 2560
    // 就是直接按照最大的seq-len分配了内存

    // 8 + 1，8 input + 1 output
    align_aside_input_bytes_ = AlignTo(max_batch_ * sizeof(int));
    whole_bytes_ = align_input_bytes_ * 4 + align_aside_input_bytes_ * 9;

    // 一次申请显存和pinned memory
    CUDA_CHECK(cudaMalloc(&d_buffer_, whole_bytes_));
    CUDA_CHECK(cudaMallocHost(&h_buffer_, whole_bytes_));

    auto h_buffer_ptr = h_buffer_;
    auto d_buffer_ptr = d_buffer_;

    device_bindings_.resize(engine_->getNbBindings());
    for (size_t i = 0; i < device_bindings_.size(); ++i) {
        device_bindings_[i] = d_buffer_ptr;
    }

    // 4 inputs [B, S]
    int b_i = 0;
    while (b_i < 4) {
        device_bindings_[start_binding_idx_ + b_i] = d_buffer_ptr;
        host_bindings_.push_back(h_buffer_ptr);
        h_buffer_ptr += align_input_bytes_;
        d_buffer_ptr += align_input_bytes_;
        b_i++;
    }

    while (b_i < 13) {
        device_bindings_[start_binding_idx_ + b_i] = d_buffer_ptr;
        host_bindings_.push_back(h_buffer_ptr);
        h_buffer_ptr += align_aside_input_bytes_;
        d_buffer_ptr += align_aside_input_bytes_;
        b_i++;
    }

    std::vector<int> input_dim = {max_batch_, max_seq_len_, 1};
    std::vector<int> aside_input_dim = {max_batch_, 1, 1};

    int binding_idx = start_binding_idx_;
    std::vector<std::vector<int>> input_dims = {input_dim, input_dim, input_dim, input_dim,
                                                aside_input_dim, aside_input_dim, aside_input_dim,aside_input_dim,
                                                aside_input_dim, aside_input_dim, aside_input_dim, aside_input_dim};
    // set device_bindings and setBindingDimensions
    // 为当前的context的连续的13个bindings设置dims，维度~
    // 之前这边是在动态shape的推理部分做的，现在就每一个profile的shape都是固定的了，那么直接就可以在构造函数做了，后面无需每次推理都做了
    for (size_t i = 0; i < input_dims.size(); ++i) {
        std::vector<int> dims_vec = input_dims[i];
        nvinfer1::Dims trt_dims;
        trt_dims.nbDims = static_cast<int>(dims_vec.size());
        // 这是？给拉长了？大小应该是3，回头验证下,突然理解了，这边不是数据，是dim，那就是三个int~
        memcpy(trt_dims.d, dims_vec.data(), trt_dims.nbDims * sizeof(int));
        context_->setBindingDimensions(binding_idx, trt_dims);
        binding_idx++;
    }

    if (!(context_->allInputDimensionsSpecified())) {
        std::cout << "(context_->allInputDimensionsSpecified() error)" << std::endl;
        assert(0);
        // 在认为不可能执行到的地方前加上这个断言，如果程序走到这里，那么一定是逻辑错误。其实就是一种预防性的错误检查。
    }

    // for (size_t i = 0; i < device_bindings_.size(); ++i) {
    //     s_device_bindings_.push_back(device_bindings_[i]);
    // }

    // warm up copy,但是为什么要把align_aside_input_bytes_减去呢？这样子大小不就不对了么？
    // 对的，把输出减掉~
    CUDA_CHECK(cudaMemcpyAsync(d_buffer_, h_buffer_, whole_bytes_ - align_aside_input_bytes_,
                               cudaMemcpyHostToDevice, cuda_stream_));
    cudaStreamSynchronize(cuda_stream_);
}

template<class T>
void _fill(T* ptr, int size, T v) {
    for (int i = 0; i < size; ++i) {
        ptr[i] = v;
    }
}

int TrtContext::CaptureCudaGraph() {
    if (graph_created_) {
        return 1;
    }

    // fill test inputs
    auto input_size = max_batch_ * max_seq_len_;
    // 模板采用了推导
    _fill((int*)host_bindings_[0], input_size, 1);
    _fill((int*)host_bindings_[1], input_size, 1);
    _fill((int*)host_bindings_[2], input_size, 1);
    _fill((float*)host_bindings_[3], input_size, 1.0f);  // mask is float

    _fill((int*)host_bindings_[4], max_batch_, 1);
    _fill((int*)host_bindings_[5], max_batch_, 1);
    _fill((int*)host_bindings_[6], max_batch_, 1);
    _fill((int*)host_bindings_[7], max_batch_, 1);
    _fill((int*)host_bindings_[8], max_batch_, 1);
    _fill((int*)host_bindings_[9], max_batch_, 1);
    _fill((int*)host_bindings_[10], max_batch_, 1);
    _fill((int*)host_bindings_[11], max_batch_, 1);

    CUDA_CHECK(cudaMemcpyAsync(d_buffer_, h_buffer_, whole_bytes_ - align_aside_input_bytes_,
                               cudaMemcpyHostToDevice, cuda_stream_));

    // warm up and let mContext do cublas initialization
    auto status = context_->enqueueV2((void**)device_bindings_.data(), cuda_stream_, nullptr);
    // TODO
    //我这边有个问题，这个s_device_bindings_到底干啥的？
    if (!status) {
        std::cerr << "enqueue failed!\n";
        exit(-1);
    }

    // enum __device_builtin__ cudaStreamCaptureMode {
    //     cudaStreamCaptureModeGlobal      = 0,
    //     cudaStreamCaptureModeThreadLocal = 1,
    //     cudaStreamCaptureModeRelaxed     = 2
    // };
    // 不晓得这几个有什么区别了~课件上的是cudaStreamCaptureModeGlobal，研究下回头~
    // TODO
    CUDA_CHECK(cudaStreamBeginCapture(cuda_stream_, cudaStreamCaptureModeRelaxed));

    status = context_->enqueueV2((void**)device_bindings_.data(), cuda_stream_, nullptr);
    if (!status) {
        std::cerr << "enqueue failed!\n";
        exit(-1);
    }

    CUDA_CHECK(cudaStreamEndCapture(cuda_stream_, &graph_));
    cudaStreamSynchronize(cuda_stream_);
    CUDA_CHECK(cudaGraphInstantiate(&instance_, graph_, NULL, NULL, 0));
    // 这边拷贝回去并未捕捉，但是上课的时候说能捕捉到，是真的么？
    // TODO
    CUDA_CHECK(cudaMemcpyAsync(host_bindings_[12], device_bindings_[12], align_aside_input_bytes_,
                               cudaMemcpyDeviceToHost, cuda_stream_));
    graph_created_ = true;
    std::cout << "profile idx: " << profile_idx_ << " CaptureCudaGraph completed!" << std::endl;
    return 0;
}

int TrtContext::Forward(sample &s) {
    cudaSetDevice(dev_id_);
    int idx = 0;
    auto batch = s.shape_info_0[0];
    auto seq_len = s.shape_info_0[1];
    auto input_bytes = batch * seq_len * sizeof(int);
    auto aside_input_bytes = batch * sizeof(int);
    memcpy(host_bindings_[0], s.i0.data(), input_bytes);
    memcpy(host_bindings_[1], s.i1.data(), input_bytes);
    memcpy(host_bindings_[2], s.i2.data(), input_bytes);
    memcpy(host_bindings_[3], s.i3.data(), input_bytes);

    memcpy(host_bindings_[4], s.i4.data(), aside_input_bytes);
    memcpy(host_bindings_[5], s.i5.data(), aside_input_bytes);
    memcpy(host_bindings_[6], s.i6.data(), aside_input_bytes);
    memcpy(host_bindings_[7], s.i7.data(), aside_input_bytes);
    memcpy(host_bindings_[8], s.i8.data(), aside_input_bytes);
    memcpy(host_bindings_[9], s.i9.data(), aside_input_bytes);
    memcpy(host_bindings_[10], s.i10.data(), aside_input_bytes);
    memcpy(host_bindings_[11], s.i11.data(), aside_input_bytes);

    cudaEvent_t start, stop;
    float elapsed_time = 0.0f;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    cudaEventQuery(start);
    CUDA_CHECK(cudaMemcpyAsync(d_buffer_, h_buffer_, whole_bytes_ - align_aside_input_bytes_,
                               cudaMemcpyHostToDevice, cuda_stream_));
    cudaStreamSynchronize(cuda_stream_);
    if (graph_created_) {
        CUDA_CHECK(cudaGraphLaunch(instance_, cuda_stream_));
    } else {
        auto status = context_->enqueueV2((void**)device_bindings_.data(), cuda_stream_, nullptr);
        if (!status) {
            std::cerr << "enqueue failed!\n";
            exit(-1);
        }
    }

    s.out_data.resize(batch);
    // trick,之前捕捉的时候是13个bindings里面偏移，但是实际推理了，就不是device_bindings_[12]了，而是要加上start_binding_idx_
    // 同时，大小也不再是是align_aside_input_bytes_，而是batch * sizeof(float)，但我感觉二者是不是一样大的
    CUDA_CHECK(cudaMemcpyAsync(s.out_data.data(), device_bindings_[start_binding_idx_ + 12], batch * sizeof(float),
                               cudaMemcpyDeviceToHost, cuda_stream_));
    cudaStreamSynchronize(cuda_stream_);
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    std::cout << "batch = " << max_batch_ << ", seq_len = " << max_seq_len_ << ", enqueue time = " << elapsed_time
              << " ms" << std::endl;
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    struct timeval tv;
    gettimeofday(&tv, NULL);
    s.timestamp = tv.tv_sec * 1000000 + tv.tv_usec;
    return 0;
}

TrtEngine::~TrtEngine(){

}

TrtContext::~TrtContext() {

}

// } // BEGIN_LIB_NAMESPACE

