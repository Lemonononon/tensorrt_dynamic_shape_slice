#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <string>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>

#include "logger.hpp"

const static char* kInputTensorName = "input";
const static char* kOutputTensorName = "output";

constexpr static int kInputH = 960;
constexpr static int kInputW = 960;

#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)\
    {\
        cudaError_t error_code = callstr;\
        if (error_code != cudaSuccess) {\
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__;\
            assert(0);\
        }\
    }
#endif  // CUDA_CHECK

using namespace nvinfer1;
static Logger gLogger;

int main(){

    std::string engine_path = "../weights/test.engine";
    bool need_build = false;
    need_build = true;

    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;
    IExecutionContext* context = nullptr;

    if (need_build){


        IBuilder* builder = createInferBuilder(gLogger);
        IBuilderConfig* config = builder->createBuilderConfig();


        INetworkDefinition* network = builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));

        ITensor* data = network->addInput(kInputTensorName, DataType::kFLOAT, Dims4{-1, 3, kInputH, kInputW});

        auto sliceLayer = network->addSlice(*data, Dims4{0, 0, 0, 0}, Dims4{-1, 3, kInputH / 2, kInputW / 2}, Dims4{1, 1, 2, 2});

        // auto sliceSize = network->getInput(0)->getDimensions();
        auto shape = network->addShape(*network->getInput(0))->getOutput(0);

        auto shapeInt32Layer = network->addIdentity(*shape);
        shapeInt32Layer->setOutputType(0, DataType::kINT32);
        auto shapeInt32 = shapeInt32Layer->getOutput(0);


        int32_t subSliceValue[4] = {0, 0, kInputH/2, kInputW/2};
        Weights subSliceWeight{DataType::kINT32, subSliceValue, 4};

        auto constLayer = network->addConstant(Dims{1, {4}}, subSliceWeight);

        auto elementLayer = network->addElementWise(*shapeInt32, *constLayer->getOutput(0), ElementWiseOperation::kSUB);

        auto newShape = elementLayer->getOutput(0);

        sliceLayer->setInput(2, *newShape);
        sliceLayer->getOutput(0)->setName(kOutputTensorName);

        network->markOutput(*sliceLayer->getOutput(0));

        auto profile = builder->createOptimizationProfile();

        profile->setDimensions(kInputTensorName, OptProfileSelector::kMIN, Dims4{1, 3, kInputH, kInputW});
        profile->setDimensions(kInputTensorName, OptProfileSelector::kOPT, Dims4{4, 3, kInputH, kInputW});
        profile->setDimensions(kInputTensorName, OptProfileSelector::kMAX, Dims4{16, 3, kInputH, kInputW});

        // builder->setMaxBatchSize(maxBatchSize);
        config->addOptimizationProfile(profile);
        config->setFlag(BuilderFlag::kFP16);
        auto engine_data = builder->buildSerializedNetwork(*network, *config);

        std::ofstream ofs(engine_path, std::ios::binary);
        if (!ofs){
            std::cout << "could not open " << engine_path << std::endl;
            return -1;
        }

        ofs.write((const char*)engine_data->data(), engine_data->size());

        runtime = createInferRuntime(gLogger);
        engine = runtime->deserializeCudaEngine(engine_data->data(), engine_data->size());
        context = engine->createExecutionContext();

    }

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    int batchSize = 4;

    // 输入输出绑定
    void* buffers[2]{};

    TensorIOMode input_mode = engine->getTensorIOMode(kInputTensorName);
    if (input_mode != TensorIOMode::kINPUT) {
        std::cerr << kInputTensorName << " should be input tensor" << std::endl;
        assert(false);
    }
    TensorIOMode output_mode = engine->getTensorIOMode(kOutputTensorName);
    if (output_mode != TensorIOMode::kOUTPUT) {
        std::cerr << kOutputTensorName << " should be output tensor" << std::endl;
        assert(false);
    }

    int OUTPUT_SIZE = 3*kInputH/2*kInputW/2;

    CUDA_CHECK(cudaMalloc(&buffers[0], batchSize * 3 * kInputH * kInputW * sizeof(float))); //对gpu进行显存分配
    CUDA_CHECK(cudaMalloc(&buffers[1], batchSize * OUTPUT_SIZE * sizeof(float)));

    context->setInputTensorAddress(kInputTensorName, buffers[0]);
    context->setOutputTensorAddress(kOutputTensorName, buffers[1]);

    std::vector<float> data;
    data.resize(batchSize * 3 * kInputH * kInputW);
    for (int i = 0; i < batchSize * 3 * kInputH * kInputW; ++i) {
        data[i] = 0.5;
    }

    std::vector<float> prob;
    prob.resize(batchSize * OUTPUT_SIZE);

    cudaMemcpyAsync(buffers[0], data.data(), batchSize * 3 * kInputH * kInputW * sizeof(float), cudaMemcpyHostToDevice, stream );
    // context->setBindingDimensions(0, nvinfer1::Dims4(1, 3, inputHeightDetection, inputWidthDetection));
    context->setInputShape(kInputTensorName, nvinfer1::Dims4(batchSize, 3, kInputH, kInputW));
    // context->enqueueV2(buffers, stream, nullptr);
    context->enqueueV3(stream);

    CUDA_CHECK(cudaMemcpyAsync(prob.data(), buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream ));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::cout << prob[0] << " " << prob[1] << std::endl;


}