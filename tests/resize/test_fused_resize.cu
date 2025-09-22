/* Copyright 2023-2025 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */


#include "tests/main.h"

#include "tests/testsCommon.cuh"
#include <opencv2/opencv.hpp>
#include <fused_kernel/algorithms/image_processing/raw_image.h>
#include <cvGPUSpeedup.cuh>

struct PerPlaneSequenceSelector {
    FK_HOST_DEVICE_FUSE uint at(const uint& index) {
        return 1;
    }
};

void testComputeWhatYouSeePlusHorizontalFusion(char* buffer, const uint& NUM_ELEMS_X, const uint& NUM_ELEMS_Y) {

    cudaStream_t stream;
    gpuErrchk(cudaStreamCreate(&stream));
    fk::Stream fk_stream{stream};

    constexpr fk::Size down(1920, 1080);
    cv::Mat h_result(down.height, down.width, CV_8UC4);
    cv::Mat nv12Image(cv::Size(NUM_ELEMS_X, NUM_ELEMS_Y + (NUM_ELEMS_Y / 2)), CV_8UC1, buffer);

    uchar* d_dataSource;
    size_t sourcePitch;
    gpuErrchk(cudaMallocPitch(&d_dataSource, &sourcePitch, NUM_ELEMS_X, NUM_ELEMS_Y + (NUM_ELEMS_Y / 2)));
    fk::RawImage<fk::PixelFormat::NV12> d_nv12Image{ {d_dataSource, {NUM_ELEMS_X, NUM_ELEMS_Y + (NUM_ELEMS_Y / 2), (uint)sourcePitch}}, NUM_ELEMS_X, NUM_ELEMS_Y};
    fk::Ptr2D<uchar4> d_rgbaImage(down.width, down.height);
    fk::Ptr2D<uchar4> d_rgbaImageBig(NUM_ELEMS_X, NUM_ELEMS_Y);

    gpuErrchk(cudaMemcpy2DAsync(d_nv12Image.data.data, d_nv12Image.data.dims.pitch,
        nv12Image.data, nv12Image.step,
        NUM_ELEMS_X, NUM_ELEMS_Y + (NUM_ELEMS_Y / 2), cudaMemcpyHostToDevice, stream));
    constexpr int CAMERAS = 4;
    constexpr int OUTPUTS = 1;
    for (int i = 0; i < CAMERAS; i++) {
        fk::Read<fk::ReadYUV<fk::PixelFormat::NV12>> read{ {d_nv12Image}};
        fk::Unary<fk::ConvertYUVToRGB<fk::PixelFormat::NV12, fk::ColorRange::Full, fk::ColorPrimitives::bt601, true>> cvtColor{};
        fk::Write<fk::PerThreadWrite<fk::ND::_2D, uchar4>> write{ d_rgbaImageBig.ptr() };
        fk::executeOperations<fk::TransformDPP<>>(fk_stream, read, cvtColor, write);

        fk::Read<fk::PerThreadRead<fk::ND::_2D, uchar4>> read2{ {d_rgbaImageBig.ptr()}};
        fk::Unary<fk::VectorReorder<uchar4, 2, 1, 0, 3>> cvtColor2{};
        fk::Write<fk::PerThreadWrite<fk::ND::_2D, uchar4>> write2{ d_rgbaImageBig.ptr() };
        fk::executeOperations<fk::TransformDPP<>>(fk_stream, read2, cvtColor2, write2);
    }

    for (int i = 0; i < OUTPUTS; i++) {
        auto read3 = fk::Resize<fk::InterpolationType::INTER_LINEAR>::build(d_rgbaImageBig.ptr(), down, 0., 0.);
        fk::Unary<fk::SaturateCast<float4, uchar4>> convertTo3{};
        fk::Write<fk::PerThreadWrite<fk::ND::_2D, uchar4>> write3{ d_rgbaImage.ptr() };
        fk::executeOperations<fk::TransformDPP<>>(fk_stream, read3, convertTo3, write3);
    }

    gpuErrchk(cudaMemcpy2DAsync(h_result.data, h_result.step,
        d_rgbaImage.ptr().data, d_rgbaImage.dims().pitch,
        down.width * sizeof(uchar4), down.height, cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaStreamSynchronize(stream));

    const auto readBackOp = fk::fuse(fk::Read<fk::ReadYUV<fk::PixelFormat::NV12>>{d_nv12Image},
                                       fk::Unary<fk::ConvertYUVToRGB<fk::PixelFormat::NV12, fk::ColorRange::Full, fk::ColorPrimitives::bt709, true, float4>>{});
    const fk::Size srcSize(NUM_ELEMS_X, NUM_ELEMS_Y);
    const auto readOp =
        fk::Resize<fk::InterpolationType::INTER_LINEAR>::build(readBackOp, down);
    auto convertOp = fk::Unary<fk::SaturateCast<float4, uchar4>>{};
    auto colorConvert = fk::Unary<fk::VectorReorder<uchar4, 2, 1, 0, 3>>{};

    fk::Write<fk::TensorWrite<uchar4>> writesTensor;
    fk::Tensor<uchar4> myTensor(down.width, down.height, OUTPUTS);
    writesTensor.params = myTensor;

    auto OpSeqTensor = fk::buildOperationSequence(readOp, convertOp, colorConvert, writesTensor);

    dim3 block = dim3(32,8);
    dim3 grid((uint)ceil((float)down.width / (float)block.x),
              (uint)ceil((float)down.height / (float)block.y),
              (uint)OUTPUTS);

    const typename fk::DivergentBatchTransformDPP<fk::ParArch::GPU_NVIDIA, PerPlaneSequenceSelector>::DPPDetails details{};

    fk::launchDivergentBatchTransformDPP_Kernel<fk::ParArch::GPU_NVIDIA, PerPlaneSequenceSelector><<<grid, block, 0, stream>>>(details, OpSeqTensor);
   
    gpuErrchk(cudaStreamSynchronize(stream));

    gpuErrchk(cudaFree(d_dataSource));

    gpuErrchk(cudaStreamDestroy(stream));
}

void testComputeWhatYouSee(char* buffer, const uint& NUM_ELEMS_X, const uint& NUM_ELEMS_Y) {

    cudaStream_t stream;
    gpuErrchk(cudaStreamCreate(&stream));
    fk::Stream fk_stream{ stream };

    constexpr fk::Size down(1920, 1080);
    cv::Mat h_result(down.height, down.width, CV_8UC4);
    cv::Mat nv12Image(cv::Size(NUM_ELEMS_X, NUM_ELEMS_Y + (NUM_ELEMS_Y / 2)), CV_8UC1, buffer);

    uchar* d_dataSource;
    size_t sourcePitch;
    gpuErrchk(cudaMallocPitch(&d_dataSource, &sourcePitch, NUM_ELEMS_X, NUM_ELEMS_Y + (NUM_ELEMS_Y / 2)));
    fk::RawImage<fk::PixelFormat::NV12> d_nv12Image{ {d_dataSource, { NUM_ELEMS_X, NUM_ELEMS_Y + (NUM_ELEMS_Y / 2), (uint)sourcePitch}}, NUM_ELEMS_X, NUM_ELEMS_Y };
    fk::Ptr2D<uchar4> d_rgbaImage(down.width, down.height);
    fk::Ptr2D<uchar4> d_rgbaImageBig(NUM_ELEMS_X, NUM_ELEMS_Y);

    gpuErrchk(cudaMemcpy2DAsync(d_nv12Image.data.data, d_nv12Image.data.dims.pitch,
                                nv12Image.data, nv12Image.step,
                                NUM_ELEMS_X, NUM_ELEMS_Y + (NUM_ELEMS_Y / 2), cudaMemcpyHostToDevice, stream));

    fk::Read<fk::ReadYUV<fk::PixelFormat::NV12>> read{ d_nv12Image };
    fk::Unary<fk::ConvertYUVToRGB<fk::PixelFormat::NV12, fk::ColorRange::Full, fk::ColorPrimitives::bt601, true>> cvtColor{};
    fk::Write<fk::PerThreadWrite<fk::ND::_2D, uchar4>> write{ d_rgbaImageBig.ptr() };
    fk::executeOperations<fk::TransformDPP<>>(fk_stream, read, cvtColor, write);

    fk::Read<fk::PerThreadRead<fk::ND::_2D, uchar4>> read2{ d_rgbaImageBig.ptr() };
    fk::Unary<fk::VectorReorder<uchar4, 2, 1, 0, 3>> cvtColor2{};
    fk::Write<fk::PerThreadWrite<fk::ND::_2D, uchar4>> write2{ d_rgbaImageBig.ptr() };
    fk::executeOperations<fk::TransformDPP<>>(fk_stream, read2, cvtColor2, write2);

    auto read3 = fk::Resize<fk::InterpolationType::INTER_LINEAR>::build(d_rgbaImageBig.ptr(), down, 0., 0.);
    fk::Unary<fk::SaturateCast<float4, uchar4>> convertTo3{};
    fk::Write<fk::PerThreadWrite<fk::ND::_2D, uchar4>> write3{ d_rgbaImage.ptr() };
    fk::executeOperations<fk::TransformDPP<>>(fk_stream, read3, convertTo3, write3);

    gpuErrchk(cudaMemcpy2DAsync(h_result.data, h_result.step,
        d_rgbaImage.ptr().data, d_rgbaImage.dims().pitch,
        down.width * sizeof(uchar4), down.height, cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaStreamSynchronize(stream));

    const auto readOpInstance = fk::fuse(fk::Read<fk::ReadYUV<fk::PixelFormat::NV12>>{d_nv12Image},
                                           fk::Unary<fk::ConvertYUVToRGB<fk::PixelFormat::NV12, fk::ColorRange::Full, fk::ColorPrimitives::bt709, true, float4>>{});
    const auto readOp = fk::Resize<fk::InterpolationType::INTER_LINEAR>::build(readOpInstance, down);
    auto convertOp = fk::Unary<fk::SaturateCast<float4, uchar4>>{};
    auto colorConvert = fk::Unary<fk::VectorReorder<uchar4, 2, 1, 0, 3>>{};
    auto writeOp = fk::Write<fk::PerThreadWrite<fk::ND::_2D, uchar4>>{ d_rgbaImage.ptr() };
    fk::executeOperations<fk::TransformDPP<>>(fk_stream, readOp, convertOp, colorConvert, writeOp);
    gpuErrchk(cudaMemcpy2DAsync(h_result.data, h_result.step,
        d_rgbaImage.ptr().data, d_rgbaImage.dims().pitch,
        down.width * sizeof(uchar4), down.height, cudaMemcpyDeviceToHost, stream));

    gpuErrchk(cudaStreamSynchronize(stream));

    gpuErrchk(cudaFree(d_dataSource));

    gpuErrchk(cudaStreamDestroy(stream));
}

int launch() {
    int returnValue = 0;

    cv::cuda::Stream cv_stream;

    cv::Mat::setDefaultAllocator(cv::cuda::HostMem::getAllocator(cv::cuda::HostMem::AllocType::PAGE_LOCKED));

    const std::string filePath{ "" };
    std::ifstream file(filePath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (file.is_open()) {
        char* buffer = new char[size];
        file.read(buffer, size);
        constexpr uint NUM_ELEMS_X = 7680;
        constexpr uint NUM_ELEMS_Y = 4320;
        testComputeWhatYouSee(buffer, NUM_ELEMS_X, NUM_ELEMS_Y);
        delete buffer;
    } else {
        // Print an error message if the file cannot be opened
        std::cout << "Cannot open file, using dummy image." << std::endl;
        constexpr uint NUM_ELEMS_X = 7680;
        constexpr uint NUM_ELEMS_Y = 4320;
        char* buffer = new char[NUM_ELEMS_X * (NUM_ELEMS_Y + (NUM_ELEMS_Y / 2))];
        testComputeWhatYouSee(buffer, NUM_ELEMS_X, NUM_ELEMS_Y);
    }
    file.close();

    const std::string filePath2{ "" };
    std::ifstream file2(filePath2, std::ios::binary | std::ios::ate);
    std::streamsize size2 = file2.tellg();
    file2.seekg(0, std::ios::beg);

    if (file2.is_open()) {
        char* buffer = new char[size2];
        file2.read(buffer, size2);
        constexpr uint NUM_ELEMS_X = 3840;
        constexpr uint NUM_ELEMS_Y = 2160;
        testComputeWhatYouSeePlusHorizontalFusion(buffer, NUM_ELEMS_X, NUM_ELEMS_Y);
        delete buffer;
    } else {
        constexpr uint NUM_ELEMS_X = 3840;
        constexpr uint NUM_ELEMS_Y = 2160;
        // Print an error message if the file cannot be opened
        std::cout << "Cannot open file, using dummy image." << std::endl;
        char* buffer = new char[NUM_ELEMS_X * (NUM_ELEMS_Y + (NUM_ELEMS_Y / 2))];
        testComputeWhatYouSeePlusHorizontalFusion(buffer, NUM_ELEMS_X, NUM_ELEMS_Y);
    }
    file2.close();

    return returnValue;
}