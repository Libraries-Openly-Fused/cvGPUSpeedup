/* Copyright 2023 Mediaproduccion S.L.U. (Oscar Amoros Huguet)
   Copyright 2025-2026 Oscar Amoros Huguet

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
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/core/utils/utils.h>
#include <fused_kernel/core/utils/vector_utils.h>
#include <fused_kernel/core/execution_model/data_parallel_patterns.h>
#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/core/data/ptr_utils.h>
#include <cvGPUSpeedup.cuh>

template <uint WIDTH, uint HEIGHT, uint BATCH, int ITERS, int IT, int OT>
bool testCircularTensorcvGS() {
    using TensorOT = typename fk::VectorTraits<CUDA_T(OT)>::base;
    constexpr uint COLOR_PLANES = CV_MAT_CN(IT);

    cvGS::CircularTensor<IT, CV_MAT_DEPTH(OT), COLOR_PLANES, BATCH, fk::CircularTensorOrder::NewestFirst> myTensor(WIDTH, HEIGHT);
    fk::Tensor<TensorOT> h_myTensor(WIDTH, HEIGHT, BATCH, COLOR_PLANES, fk::MemType::HostPinned);
    cv::cuda::GpuMat input(HEIGHT, WIDTH, IT);
    fk::Ptr2D<CUDA_T(IT)> h_input(WIDTH, HEIGHT, 0, fk::MemType::HostPinned);

    fk::Stream fk_stream;

    fk::setTo(10.f, h_myTensor, fk_stream);

    cv::cuda::Stream cv_stream = cv::cuda::StreamAccessor::wrapStream(fk_stream);

    gpuErrchk(cudaMemcpyAsync(myTensor.ptr().data, h_myTensor.ptr().data, myTensor.sizeInBytes(), cudaMemcpyHostToDevice, fk_stream));

    for (int i = 0; i < ITERS; i++) {
        fk::setTo(fk::make_<CUDA_T(IT)>(i + 1, i + 1, i + 1), h_input, fk_stream);
        gpuErrchk(cudaMemcpy2DAsync(input.data, input.step,
                                    h_input.ptr().data, h_input.ptr().dims.pitch,
                                    h_input.ptr().dims.width * sizeof(CUDA_T(IT)),
                                    h_input.ptr().dims.height,
                                    cudaMemcpyHostToDevice, fk_stream));
        myTensor.update(cv_stream, input,
                        fk::Unary<fk::SaturateCast<CUDA_T(IT), CUDA_T(OT)>> {},
                        fk::Write<fk::TensorSplit<CUDA_T(OT)>> {myTensor.ptr()});
        fk_stream.sync();
    }

    gpuErrchk(cudaMemcpyAsync(h_myTensor.ptr().data, myTensor.ptr().data, myTensor.sizeInBytes(), cudaMemcpyDeviceToHost, fk_stream));

    fk_stream.sync();

    bool correct = true;
    const size_t plane_pixels = h_myTensor.dims().width * h_myTensor.dims().height;
    for (int z = 0; z < BATCH; z++) {
        const TensorOT value = (TensorOT)(ITERS - z);
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                const fk::Point p{x, y, z};
                const TensorOT* workPlane = fk::PtrAccessor<fk::ND::_3D>::point(p, h_myTensor.ptr());
                const TensorOT resX = *workPlane;
                correct &= value == resX;
                const TensorOT resY = *(workPlane + plane_pixels);
                correct &= value == resY;
                const TensorOT resZ = *(workPlane + (plane_pixels * 2));
                correct &= value == resZ;
            }
        }
    }

    return correct;
}

template <int IT, int OT>
bool testTransposedCircularTensorcvGS() {
    using TensorOT = typename fk::VectorTraits<CUDA_T(OT)>::base;
    constexpr uint BATCH = 15;
    constexpr uint WIDTH = 128;
    constexpr uint HEIGHT = 128;
    constexpr uint COLOR_PLANES = CV_MAT_CN(IT);
    constexpr int ITERS = 100;

    cvGS::CircularTensor<IT, CV_MAT_DEPTH(OT), COLOR_PLANES, BATCH, fk::CircularTensorOrder::NewestFirst, fk::ColorPlanes::Transposed> myTensor(WIDTH, HEIGHT);
    fk::TensorT<TensorOT> h_myTensor(WIDTH, HEIGHT, BATCH, COLOR_PLANES, fk::MemType::HostPinned);
    fk::TensorT<TensorOT> h_myInternalTensor(WIDTH, HEIGHT, BATCH, COLOR_PLANES, fk::MemType::HostPinned);
    cv::cuda::GpuMat input(HEIGHT, WIDTH, IT);
    fk::Ptr2D<CUDA_T(IT)> h_input(WIDTH, HEIGHT, 0, fk::MemType::HostPinned);

    fk::Stream stream;
    fk::setTo(10.f, h_myTensor, stream);

    cv::cuda::Stream cv_stream = cv::cuda::StreamAccessor::wrapStream(stream);

    gpuErrchk(cudaMemcpyAsync(myTensor.ptr().data, h_myTensor.ptr().data, myTensor.sizeInBytes(), cudaMemcpyHostToDevice, stream));

    for (int i = 0; i < ITERS; i++) {
        fk::setTo(fk::make_<CUDA_T(IT)>(i + 1, i + 1, i + 1), h_input, stream);
        gpuErrchk(cudaMemcpy2DAsync(input.data, input.step,
            h_input.ptr().data, h_input.ptr().dims.pitch,
            h_input.ptr().dims.width * sizeof(CUDA_T(IT)),
            h_input.ptr().dims.height,
            cudaMemcpyHostToDevice, stream));
        myTensor.update(cv_stream, input,
            fk::Unary<fk::SaturateCast<CUDA_T(IT), CUDA_T(OT)>> {},
            fk::Write<fk::TensorTSplit<CUDA_T(OT)>> {myTensor.ptr()});
        gpuErrchk(cudaStreamSynchronize(stream));
    }

    gpuErrchk(cudaMemcpyAsync(h_myTensor.ptr().data, myTensor.ptr().data, myTensor.sizeInBytes(), cudaMemcpyHostToDevice, stream));

    gpuErrchk(cudaStreamSynchronize(stream));

    bool correct = true;
    const auto dims = h_myTensor.dims();
    const size_t plane_pixels = dims.width * dims.height;
    for (int cp = 0; cp < (int)dims.color_planes; cp++) {
        for (int y = 0; y < (int)dims.height; y++) {
            for (int z = 0; z < (int)BATCH; z++) {
                const auto* plane = fk::PtrAccessor<fk::ND::T3D>::cr_point(fk::Point{0, 0, z}, h_myTensor.ptr())
                    + (plane_pixels * dims.planes * cp);
                for (int x = 0; x < (int)dims.width; x++) {
                    correct &= ITERS - z == plane[x + (y * dims.width)];
                }
            }
        }
    }

    return correct;
}

template <int IT, int OT>
bool testTransposedOldestFirstCircularTensorcvGS() {
    using TensorOT = typename fk::VectorTraits<CUDA_T(OT)>::base;
    constexpr uint BATCH = 15;
    constexpr uint WIDTH = 128;
    constexpr uint HEIGHT = 128;
    constexpr uint COLOR_PLANES = CV_MAT_CN(IT);
    constexpr int ITERS = 100;

    cvGS::CircularTensor<IT, CV_MAT_DEPTH(OT), COLOR_PLANES, BATCH, fk::CircularTensorOrder::OldestFirst, fk::ColorPlanes::Transposed> myTensor(WIDTH, HEIGHT);
    fk::TensorT<TensorOT> h_myTensor(WIDTH, HEIGHT, BATCH, COLOR_PLANES, fk::MemType::HostPinned);
    fk::TensorT<TensorOT> h_myInternalTensor(WIDTH, HEIGHT, BATCH, COLOR_PLANES, fk::MemType::HostPinned);
    cv::cuda::GpuMat input(HEIGHT, WIDTH, IT);
    fk::Ptr2D<CUDA_T(IT)> h_input(WIDTH, HEIGHT, 0, fk::MemType::HostPinned);

    fk::Stream stream;

    fk::setTo(10.f, h_myTensor, stream);

    cv::cuda::Stream cv_stream = cv::cuda::StreamAccessor::wrapStream(stream);

    gpuErrchk(cudaMemcpyAsync(myTensor.ptr().data, h_myTensor.ptr().data, myTensor.sizeInBytes(), cudaMemcpyHostToDevice, stream));

    for (int i = 0; i < ITERS; i++) {
        fk::setTo(fk::make_<CUDA_T(IT)>(i + 1, i + 1, i + 1), h_input, stream);
        gpuErrchk(cudaMemcpy2DAsync(input.data, input.step,
            h_input.ptr().data, h_input.ptr().dims.pitch,
            h_input.ptr().dims.width * sizeof(CUDA_T(IT)),
            h_input.ptr().dims.height,
            cudaMemcpyHostToDevice, stream));
        myTensor.update(cv_stream, input,
            fk::Unary<fk::SaturateCast<CUDA_T(IT), CUDA_T(OT)>> {},
            fk::Write<fk::TensorTSplit<CUDA_T(OT)>> {myTensor.ptr()});
        gpuErrchk(cudaStreamSynchronize(stream));
    }

    gpuErrchk(cudaMemcpyAsync(h_myTensor.ptr().data, myTensor.ptr().data, myTensor.sizeInBytes(), cudaMemcpyHostToDevice, stream));

    gpuErrchk(cudaStreamSynchronize(stream));

    bool correct = true;
    const auto dims = h_myTensor.dims();
    const size_t plane_pixels = dims.width * dims.height;
    for (int cp = 0; cp < (int)dims.color_planes; cp++) {
        for (int y = 0; y < (int)dims.height; y++) {
            for (int z = 0; z < (int)BATCH; z++) {
                const auto* plane = fk::PtrAccessor<fk::ND::T3D>::cr_point(fk::Point{0, 0, z}, h_myTensor.ptr())
                    + (plane_pixels * dims.planes * cp);
                for (int x = 0; x < (int)dims.width; x++) {
                    correct &= ITERS - (BATCH - z - 1) == plane[x + (y * dims.width)];
                }
            }
        }
    }

    return correct;
}

bool testOldestFirstCircularTensorcvGS_noSplit() {
    constexpr uint BATCH = 15;
    constexpr uint WIDTH = 128;
    constexpr uint HEIGHT = 128;
    // Number of planes representing one image
    constexpr uint COLOR_PLANES = 1; // This means that the image is in packed mode, each data element will contain all the color chanels for the same pixel
    constexpr int ITERS = 100;

    cvGS::CircularTensor<CV_8UC4, CV_32FC4, COLOR_PLANES, BATCH, fk::CircularTensorOrder::OldestFirst> myTensor(WIDTH, HEIGHT);
    using TensorType = CUDA_T(CV_32FC4);
    fk::Tensor<TensorType> h_myTensor(WIDTH, HEIGHT, BATCH, COLOR_PLANES, fk::MemType::HostPinned);
    fk::Tensor<TensorType> h_myInternalTensor(WIDTH, HEIGHT, BATCH, COLOR_PLANES, fk::MemType::HostPinned);
    cv::cuda::GpuMat input(HEIGHT, WIDTH, CV_8UC4);
    fk::Ptr2D<CUDA_T(CV_8UC4)> h_input(WIDTH, HEIGHT, 0, fk::MemType::HostPinned);

    cudaStream_t stream;
    gpuErrchk(cudaStreamCreate(&stream));
    fk::Stream fk_stream(stream);

    fk::setTo(fk::make_set<float4>(10.0f), h_myTensor, fk_stream);

    cv::cuda::Stream cv_stream = cv::cuda::StreamAccessor::wrapStream(stream);

    gpuErrchk(cudaMemcpyAsync(myTensor.ptr().data, h_myTensor.ptr().data, myTensor.sizeInBytes(), cudaMemcpyHostToDevice, stream));

    for (int i = 0; i < ITERS; i++) {
        fk::setTo(fk::make_set<CUDA_T(CV_8UC4)>(i + 1), h_input, fk_stream);
        gpuErrchk(cudaMemcpy2DAsync(input.data, input.step,
                                    h_input.ptr().data, h_input.ptr().dims.pitch,
                                    h_input.ptr().dims.width * sizeof(CUDA_T(CV_8UC4)),
                                    h_input.ptr().dims.height,
                                    cudaMemcpyHostToDevice, stream));
        myTensor.update(cv_stream, input,
                        fk::Unary<fk::SaturateCast<CUDA_T(CV_8UC4), CUDA_T(CV_32FC4)>> {},
                        fk::Write<fk::TensorWrite<CUDA_T(CV_32FC4)>> {myTensor.ptr()});
                        gpuErrchk(cudaStreamSynchronize(stream));
    }

    gpuErrchk(cudaMemcpyAsync(h_myTensor.ptr().data, myTensor.ptr().data, myTensor.sizeInBytes(), cudaMemcpyHostToDevice, stream));

    gpuErrchk(cudaStreamSynchronize(stream));

    bool correct = true;
    const auto dims = h_myTensor.dims();
    const size_t plane_pixels = dims.width * dims.height;
    for (int cp = 0; cp < (int)dims.color_planes; cp++) {
        for (int y = 0; y < (int)dims.height; y++) {
            for (int z = 0; z < (int)BATCH; z++) {
                const float4* plane = fk::PtrAccessor<fk::ND::_3D>::cr_point(fk::Point{0, 0, z}, h_myTensor.ptr()) + (plane_pixels * dims.planes * cp);
                for (int x = 0; x < (int)dims.width; x++) {
                    const float4 groundTruth = fk::make_set<float4>(ITERS - (BATCH - z - 1));
                    const float4 computedValue = plane[x + (y * dims.width)];
                    correct &= abs(groundTruth.x - computedValue.x) < 0.00001f;
                    correct &= abs(groundTruth.y - computedValue.y) < 0.00001f;
                    correct &= abs(groundTruth.z - computedValue.z) < 0.00001f;
                    correct &= abs(groundTruth.w - computedValue.w) < 0.00001f;
                }
            }
        }
    }

    return correct;
}

template <uint WIDTH, uint HEIGHT, uint BATCH, int ITERS, int IT, int OT>
bool launchTestCircularTensorcvGS() {
    if (testCircularTensorcvGS<WIDTH, HEIGHT, BATCH, ITERS, IT, OT>()) {
        std::cout << "testCircularTensorcvGS<" << WIDTH << ", " << HEIGHT << ", " << BATCH << ", " << ITERS << ", " << IT << ", " << OT << "> OK" << std::endl;
        return true;
    } else {
        std::cout << "testCircularTensorcvGS<" << WIDTH << ", " << HEIGHT << ", " << BATCH << ", " << ITERS << ", " << IT << ", " << OT << "> Failed!" << std::endl;
        return false;
    }
}

int launch() {
    int returnValue = 0;

    bool correct{true};
    correct &= launchTestCircularTensorcvGS<128, 128, 2, 100, CV_8UC3, CV_32FC3>();
    correct &= launchTestCircularTensorcvGS<128, 128, 3, 100, CV_8UC3, CV_32FC3>();
    correct &= launchTestCircularTensorcvGS<128, 128, 4, 100, CV_8UC3, CV_32FC3>();
    correct &= launchTestCircularTensorcvGS<128, 128, 5, 100, CV_8UC3, CV_32FC3>();
    correct &= launchTestCircularTensorcvGS<128, 128, 6, 100, CV_8UC3, CV_32FC3>();
    correct &= launchTestCircularTensorcvGS<128, 128, 7, 100, CV_8UC3, CV_32FC3>();
    correct &= launchTestCircularTensorcvGS<128, 128, 8, 100, CV_8UC3, CV_32FC3>();
    correct &= launchTestCircularTensorcvGS<128, 128, 9, 100, CV_8UC3, CV_32FC3>();
    correct &= launchTestCircularTensorcvGS<128, 128, 10, 100, CV_8UC3, CV_32FC3>();
    correct &= launchTestCircularTensorcvGS<128, 128, 11, 100, CV_8UC3, CV_32FC3>();
    correct &= launchTestCircularTensorcvGS<128, 128, 12, 100, CV_8UC3, CV_32FC3>();
    correct &= launchTestCircularTensorcvGS<128, 128, 13, 100, CV_8UC3, CV_32FC3>();
    correct &= launchTestCircularTensorcvGS<128, 128, 14, 100, CV_8UC3, CV_32FC3>();
    correct &= launchTestCircularTensorcvGS<128, 128, 15, 100, CV_8UC3, CV_32FC3>();
    returnValue = correct ? returnValue : -1;

    if (testTransposedCircularTensorcvGS<CV_8UC3, CV_32FC3>()) {
        std::cout << "testTransposedCircularTensorcvGS<CV_8UC3, CV_32FC3> OK" << std::endl;
    } else {
        std::cout << "testTransposedCircularTensorcvGS <CV_8UC3, CV_32FC3> Failed!" << std::endl;
        returnValue = -1;
    }
    if (testTransposedOldestFirstCircularTensorcvGS<CV_8UC3, CV_32FC3>()) {
        std::cout << "testTransposedOldestFirstCircularTensorcvGS<CV_8UC3, CV_32FC3> OK" << std::endl;
    } else {
        std::cout << "testTransposedOldestFirstCircularTensorcvGS <CV_8UC3, CV_32FC3> Failed!" << std::endl;
        returnValue = -1;
    }
    if (testOldestFirstCircularTensorcvGS_noSplit()) {
        std::cout << "testOldestFirstCircularTensorcvGS_noSplit OK" << std::endl;
    } else {
        std::cout << "testOldestFirstCircularTensorcvGS_noSplit Failed!" << std::endl;
        returnValue = -1;
    }

    return returnValue;
}