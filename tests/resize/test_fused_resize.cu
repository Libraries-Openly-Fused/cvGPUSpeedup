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
#include <fused_kernel/algorithms/image_processing/image.h>
#include <fused_kernel/algorithms/image_processing/image_processing.h>
#include <fused_kernel/fused_kernel.h>

struct PerPlaneSequenceSelector {
    FK_HOST_DEVICE_FUSE uint at(const uint& index) {
        return 1;
    }
};

void testComputeWhatYouSeePlusHorizontalFusion(char* buffer, const uint& NUM_ELEMS_X, const uint& NUM_ELEMS_Y) {
    using namespace fk;
    Stream fk_stream;

    constexpr Size down(1920, 1080);

    Image<PixelFormat::NV12> nv12Image(NUM_ELEMS_X, NUM_ELEMS_Y);
    memcpy(nv12Image.getData().ptrPinned().data, buffer, NUM_ELEMS_X * (NUM_ELEMS_Y + (NUM_ELEMS_Y / 2)));
    Ptr2D<uchar4> rgbaImage(down.width, down.height);
    Ptr2D<uchar4> rgbaImageBig(NUM_ELEMS_X, NUM_ELEMS_Y);
    nv12Image.upload(fk_stream);

    constexpr int CAMERAS = 4;
    constexpr int OUTPUTS = 1;
    for (int i = 0; i < CAMERAS; i++) {
        const auto read = ReadYUV<PixelFormat::NV12>::build(nv12Image);
        const auto cvtColor = ConvertYUVToRGB<ColorDepth::p8bit, ColorRange::Full, ColorPrimitives::bt601>::build();
        const auto addAlpha = AddOpaqueAlpha<float3, ColorDepth::p8bit>::build();
        const auto cast = SaturateCast<float4, uchar4>::build();
        const auto write = PerThreadWrite<ND::_2D, uchar4>::build(rgbaImageBig.ptr());
        executeOperations<TransformDPP<>>(fk_stream, read, cvtColor, addAlpha, cast, write);

        const auto read2 = PerThreadRead<ND::_2D, uchar4>::build(rgbaImageBig.ptr());
        const auto cvtColor2 = VectorReorder<uchar4, 2, 1, 0, 3>::build();
        const auto write2 = PerThreadWrite<ND::_2D, uchar4>::build(rgbaImageBig.ptr());
        executeOperations<TransformDPP<>>(fk_stream, read2, cvtColor2, write2);
    }

    for (int i = 0; i < OUTPUTS; i++) {
        const auto read3 = Resize<InterpolationType::INTER_LINEAR>::build(rgbaImageBig.ptr(), down, 0., 0.);
        const auto convertTo3 = SaturateCast<float4, uchar4>::build();
        const auto write3 = PerThreadWrite<ND::_2D, uchar4>::build(rgbaImage.ptr());
        executeOperations<TransformDPP<>>(fk_stream, read3, convertTo3, write3);
    }

    fk_stream.sync();

    const auto readBackOp = ReadYUV<PixelFormat::NV12>::build(nv12Image).then(
                                       ConvertYUVToRGB<ColorDepth::p8bit, ColorRange::Full, ColorPrimitives::bt709>::build());
    const Size srcSize(NUM_ELEMS_X, NUM_ELEMS_Y);
    const auto readOp = Resize<InterpolationType::INTER_LINEAR>::build(readBackOp, down);
    const auto addAlpha = AddOpaqueAlpha<float3, ColorDepth::p8bit>::build();
    auto convertOp = SaturateCast<float4, uchar4>::build();
    auto colorConvert = VectorReorder<uchar4, 2, 1, 0, 3>::build();

    Tensor<uchar4> myTensor(down.width, down.height, OUTPUTS);
    const auto writesTensor = TensorWrite<uchar4>::build(myTensor);

    auto OpSeqTensor = buildOperationSequence(readOp, addAlpha, convertOp, colorConvert, writesTensor);

    executeOperations<DivergentBatchTransformDPP<ParArch::GPU_NVIDIA, PerPlaneSequenceSelector>>(fk_stream,
                                                                                                 OpSeqTensor);
    fk_stream.sync();
}

void testComputeWhatYouSee(char *buffer, const uint &NUM_ELEMS_X, const uint &NUM_ELEMS_Y) {
    using namespace fk;
    Stream fk_stream;

    constexpr Size down(1920, 1080);
    
    Image<PixelFormat::NV12> nv12Image(NUM_ELEMS_X, NUM_ELEMS_Y);
    memcpy(nv12Image.getData().ptrPinned().data, buffer, NUM_ELEMS_X * (NUM_ELEMS_Y + (NUM_ELEMS_Y / 2)));
    Ptr2D<uchar4> rgbaImage(down.width, down.height);
    Ptr2D<uchar4> rgbaImageBig(NUM_ELEMS_X, NUM_ELEMS_Y);
    nv12Image.upload(fk_stream);

    const auto read = ReadYUV<PixelFormat::NV12>::build(nv12Image);
    const auto cvtColor = ConvertYUVToRGB<ColorDepth::p8bit, ColorRange::Full, ColorPrimitives::bt601>::build();
    const auto addAlpha = AddOpaqueAlpha<float3, ColorDepth::p8bit>::build();
    const auto cast = SaturateCast<float4, uchar4>::build();
    const auto write = PerThreadWrite<ND::_2D, uchar4>::build(rgbaImageBig.ptr());
    executeOperations<TransformDPP<>>(fk_stream, read, cvtColor, addAlpha, cast, write);

    const auto read2 = PerThreadRead<ND::_2D, uchar4>::build(rgbaImageBig.ptr());
    const auto cvtColor2 = VectorReorder<uchar4, 2, 1, 0, 3>::build();
    const auto write2 = PerThreadWrite<ND::_2D, uchar4>::build(rgbaImageBig.ptr());
    executeOperations<TransformDPP<>>(fk_stream, read2, cvtColor2, write2);

    const auto read3 = Resize<InterpolationType::INTER_LINEAR>::build(rgbaImageBig.ptr(), down, 0., 0.);
    const auto convertTo3 = SaturateCast<float4, uchar4>::build();
    const auto write3 = PerThreadWrite<ND::_2D, uchar4>::build(rgbaImage.ptr());
    executeOperations<TransformDPP<>>(fk_stream, read3, convertTo3, write3);

    rgbaImageBig.download(fk_stream);
    rgbaImage.download(fk_stream);
    fk_stream.sync();

    const auto readOpInstance = ReadYUV<PixelFormat::NV12>::build(nv12Image).then(
                                           ConvertYUVToRGB<ColorDepth::p8bit, ColorRange::Full, ColorPrimitives::bt709>::build());
    const auto readOp = Resize<InterpolationType::INTER_LINEAR>::build(readOpInstance, down);
    const auto convertOp = SaturateCast<float4, uchar4>::build();
    const auto colorConvert = VectorReorder<uchar4, 2, 1, 0, 3>::build();
    const auto writeOp = PerThreadWrite<ND::_2D, uchar4>::build(rgbaImage.ptr());
    executeOperations<TransformDPP<>>(fk_stream, readOp, addAlpha, convertOp, colorConvert, writeOp);
    rgbaImage.download(fk_stream);
}

int launch() {
    int returnValue = 0;

    const std::string filePath{""};
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