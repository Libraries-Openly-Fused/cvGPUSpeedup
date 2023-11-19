/* Copyright 2023 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include <fstream>
#include <iostream>

#include "testsCommon.cuh"
#include <opencv2/opencv.hpp>
#include <cvGPUSpeedup.cuh>

int main() {
    constexpr size_t NUM_ELEMS_X = 6244;
    constexpr size_t NUM_ELEMS_Y = 4168;

    cv::cuda::Stream cv_stream;

    cv::Mat::setDefaultAllocator(cv::cuda::HostMem::getAllocator(cv::cuda::HostMem::AllocType::PAGE_LOCKED));

    const std::string filePath{ "C:/Users/oscar/Documents/GitHub/cvGPUSpeedup/images/raw6K.nv12" };
    // Open a binary file named "example.bin"
    std::ifstream file(filePath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    char* buffer = new char[size];
    if (file.read(buffer, size)) {
        // use buffer
        cudaStream_t stream;
        gpuErrchk(cudaStreamCreate(&stream));

        constexpr fk::Size down(1920, 1080);
        cv::Mat h_result(down.height, down.width, CV_8UC4);
        cv::Mat nv12Image(cv::Size(NUM_ELEMS_X, NUM_ELEMS_Y + (NUM_ELEMS_Y/2)), CV_8UC1, buffer);

        uchar* d_dataSource;
        size_t sourcePitch;
        gpuErrchk(cudaMallocPitch(&d_dataSource, &sourcePitch, NUM_ELEMS_X, NUM_ELEMS_Y + (NUM_ELEMS_Y / 2)));
        fk::RawPtr<fk::_2D, uchar> d_nv12Image{ d_dataSource, {(uint)NUM_ELEMS_X, (uint)NUM_ELEMS_Y, (uint)sourcePitch} };
        fk::Ptr2D<uchar4> d_rgbaImage(down.width, down.height);
        fk::Ptr2D<uchar4> d_rgbaImageBig(NUM_ELEMS_X, NUM_ELEMS_Y);

        gpuErrchk(cudaMemcpy2DAsync(d_nv12Image.data, d_nv12Image.dims.pitch,
                          nv12Image.data, nv12Image.step,
                          NUM_ELEMS_X, NUM_ELEMS_Y + (NUM_ELEMS_Y / 2), cudaMemcpyHostToDevice, stream));

        fk::Read<fk::ReadYUV<fk::NV12>> read { d_nv12Image, {NUM_ELEMS_X, NUM_ELEMS_Y} };
        fk::Unary<fk::ConvertYUVToRGB<fk::NV12, fk::Full, fk::bt709, true>> cvtColor {};
        fk::Write<fk::PerThreadWrite<fk::_2D, uchar4>> write { d_rgbaImageBig.ptr() };
        fk::executeOperations(stream, read, cvtColor, write);

        fk::Read<fk::PerThreadRead<fk::_2D, uchar4>> read2{ d_rgbaImageBig.ptr(), {NUM_ELEMS_X, NUM_ELEMS_Y} };
        fk::Unary<fk::VectorReorder<uchar4, 2, 1, 0, 3>> cvtColor2{};
        fk::Write<fk::PerThreadWrite<fk::_2D, uchar4>> write2{ d_rgbaImageBig.ptr() };
        fk::executeOperations(stream, read2, cvtColor2, write2);

        auto read3 = fk::resize<uchar4, fk::INTER_LINEAR>(d_rgbaImageBig.ptr(), down, 0., 0.);
        fk::Unary<fk::SaturateCast<float4, uchar4>> convertTo3 {};
        fk::Write<fk::PerThreadWrite<fk::_2D, uchar4>> write3 { d_rgbaImage.ptr() };
        fk::executeOperations(stream, read3, convertTo3, write3);
        gpuErrchk(cudaMemcpy2DAsync(h_result.data, h_result.step,
                                    d_rgbaImage.ptr().data, d_rgbaImage.dims().pitch,
                                    down.width * sizeof(uchar4), down.height, cudaMemcpyDeviceToHost, stream));
        gpuErrchk(cudaStreamSynchronize(stream));

        using PixelReadOp = fk::ComposedOperation<fk::Read<fk::ReadYUV<fk::NV12>>, fk::Unary<fk::ConvertYUVToRGB<fk::NV12, fk::Full, fk::bt709, true>>>;
        fk::Binary<PixelReadOp> readOpInstance{ { {d_nv12Image, {}}, {} } };
        auto imgSize = d_nv12Image.dims;
        auto readOp = fk::resize<PixelReadOp, fk::INTER_LINEAR>(readOpInstance.params, fk::Size(imgSize.width, imgSize.height), down);
        auto convertOp = fk::Unary<fk::SaturateCast<float4, uchar4>>{};
        auto colorConvert = fk::Unary<fk::VectorReorder<uchar4, 2, 1, 0, 3>>{};
        auto writeOp = fk::Write<fk::PerThreadWrite<fk::_2D, uchar4>>{ d_rgbaImage.ptr() };
        fk::executeOperations(stream, readOp, convertOp, colorConvert, writeOp);
        gpuErrchk(cudaMemcpy2DAsync(h_result.data, h_result.step,
                                    d_rgbaImage.ptr().data, d_rgbaImage.dims().pitch,
                                    down.width * sizeof(uchar4), down.height, cudaMemcpyDeviceToHost, stream));

        gpuErrchk(cudaStreamSynchronize(stream));

        gpuErrchk(cudaFree(d_dataSource));

        gpuErrchk(cudaStreamDestroy(stream));

    } else {
        // Print an error message if the file cannot be opened
        std::cerr << "Error: cannot open file\n";
    }
    file.close();
    delete buffer;

    return 0;
}