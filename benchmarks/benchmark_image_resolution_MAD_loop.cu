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
#include <cvGPUSpeedup.cuh>
#include <opencv2/cudaimgproc.hpp>
#include <fused_kernel/algorithms/basic_ops/static_loop.h>

#ifdef ENABLE_BENCHMARK

constexpr char VARIABLE_DIMENSION[]{ "Number of pixels per side" };
#ifndef CUDART_MAJOR_VERSION
#error CUDART_MAJOR_VERSION Undefined!
#elif (CUDART_MAJOR_VERSION == 11)
constexpr size_t NUM_EXPERIMENTS = 10;
constexpr size_t FIRST_VALUE = 10;
constexpr size_t INCREMENT = 100;
#elif (CUDART_MAJOR_VERSION == 12)
constexpr size_t NUM_EXPERIMENTS = 60;
constexpr size_t FIRST_VALUE = 100;
constexpr size_t INCREMENT = 282270;
#endif // CUDART_MAJOR_VERSION
constexpr std::array<size_t, NUM_EXPERIMENTS> batchValues = arrayIndexSecuence<FIRST_VALUE, INCREMENT, NUM_EXPERIMENTS>;

template <int CV_TYPE_I, int CV_TYPE_O>
struct VerticalFusionMAD {
    static inline void execute(const cv::cuda::GpuMat& cvInput,
                               cv::cuda::Stream& cv_stream,
                               const cv::Scalar& val_mul,
                               const cv::Scalar& val_add,
                               const cv::cuda::GpuMat& d_output) {
        using InputType = CUDA_T(CV_TYPE_I);
        using OutputType = CUDA_T(CV_TYPE_O);
        using Loop = fk::Binary<fk::StaticLoop<fk::FusedOperation<fk::Mul<OutputType>, fk::Add<OutputType>>, 200/2>>;

        Loop loop;
        fk::get<0>(loop.params).params = cvGS::cvScalar2CUDAV<CV_TYPE_O>::get(val_mul);
        fk::get<1>(loop.params).params = cvGS::cvScalar2CUDAV<CV_TYPE_O>::get(val_add);

        if (cvInput.rows > 1) {
            throw std::runtime_error("VerticalFusionMAD only supports 1D input data.");
        }
        const uint inputWidth = static_cast<uint>(cvInput.cols);
        const uint outputWidth = static_cast<uint>(d_output.cols);

        const fk::RawPtr<fk::ND::_1D, InputType> fkInput{ reinterpret_cast<InputType*>(cvInput.data), { inputWidth, static_cast<uint>(inputWidth * sizeof(InputType)) }};
        const auto readOp = fk::PerThreadRead<fk::ND::_1D, InputType>::build(fkInput);

        const fk::RawPtr<fk::ND::_1D, OutputType> fkOutput{ reinterpret_cast<OutputType*>(d_output.data), { outputWidth, static_cast<uint>(outputWidth * sizeof(OutputType)) } };
        const auto writeOp = fk::PerThreadWrite<fk::ND::_1D, OutputType>::build(fkOutput);

        constexpr bool THREAD_FUSION = false;
        const auto tDetails = fk::TransformDPP<fk::ParArch::GPU_NVIDIA>::build_details<THREAD_FUSION>(readOp, cvGS::convertTo<CV_TYPE_I, CV_TYPE_O>(), loop, writeOp);

        const dim3 block(256);
        const dim3 grid(ceil(inputWidth / static_cast<float>(block.x)));
        const cudaStream_t stream = cv::cuda::StreamAccessor::getStream(cv_stream);
        fk::launchTransformDPP_Kernel<fk::ParArch::GPU_NVIDIA, fk::TF::ENABLED><<<grid, block, 0, stream>>>(tDetails, readOp, cvGS::convertTo<CV_TYPE_I, CV_TYPE_O>(), loop, writeOp);
        gpuErrchk(cudaGetLastError());
    }
};

// Here BATCH means resolution
template <int CV_TYPE_I, int CV_TYPE_O, size_t BATCH>
bool benchmark_image_resolution_MAD_loop(cv::cuda::Stream& cv_stream, bool enabled) {
    constexpr size_t NUM_ELEMS_X = BATCH;
    constexpr size_t NUM_ELEMS_Y = 1;
    std::stringstream error_s;
    bool passed = true;
    bool exception = false;

    static_assert(BATCH != 0, "Must not be zero");

    if (enabled) {
        struct Parameters {
            const cv::Scalar init;
            const cv::Scalar alpha;
            const cv::Scalar val_mul;
            const cv::Scalar val_add;
        };

        double alpha = 1.0;

        const Parameters one{ {1u}, {alpha}, {1.f}, {-3.2f} };
        const Parameters two{ {1u, 2u}, {alpha, alpha}, {1.f, 4.f}, {-3.2f, -0.6f} };
        const Parameters three{ {1u, 2u, 3u}, {alpha, alpha, alpha}, {1.f, 4.f, 2.f}, {-3.2f, -0.6f, -3.8f} };
        const Parameters four{ {1u, 2u, 3u, 4u}, {alpha, alpha, alpha, alpha}, {1.f, 4.f, 2.f, 0.5f}, {-3.2f, -0.6f, -3.8f, -33.f} };
        const std::array<Parameters, 4> params{ one, two, three, four };

        const cv::Scalar val_init = params.at(CV_MAT_CN(CV_TYPE_O) - 1).init;
        const cv::Scalar val_alpha = params.at(CV_MAT_CN(CV_TYPE_O) - 1).alpha;
        const cv::Scalar val_mul = params.at(CV_MAT_CN(CV_TYPE_O) - 1).val_mul;
        const cv::Scalar val_add = params.at(CV_MAT_CN(CV_TYPE_O) - 1).val_add;
        try {
            const cv::Size dataSize(NUM_ELEMS_X, NUM_ELEMS_Y);

            cv::cuda::GpuMat cvInput(dataSize, CV_TYPE_I, val_init);
            cv::cuda::GpuMat cvOutput(dataSize, CV_TYPE_O, cv::Scalar(0));
            cv::cuda::GpuMat cvTemp(dataSize, CV_TYPE_O, cv::Scalar(0));
            cv::Mat h_cvOutput(dataSize, CV_TYPE_O, cv::Scalar(0));
            cv::cuda::GpuMat cvGSOutput(dataSize, CV_TYPE_O, cv::Scalar(0));
            cv::Mat h_cvGSOutput(dataSize, CV_TYPE_O, cv::Scalar(0));

            START_OCV_BENCHMARK
            // cvGS individual kernels
            
            cvInput.convertTo(cvOutput, CV_TYPE_O, 1, cv_stream);
            for (int numOp = 0; numOp < 200; numOp+=2) {
                cv::cuda::multiply(cvOutput, val_mul, cvTemp, 1.0, -1, cv_stream);
                cv::cuda::add(cvTemp, val_add, cvOutput, cv::noArray(), -1, cv_stream);
            }
            STOP_OCV_START_CVGS_BENCHMARK
            // cvGPUSpeedup
            VerticalFusionMAD<CV_TYPE_I, CV_TYPE_O>::execute(cvInput, cv_stream, val_mul, val_add, cvGSOutput);
            STOP_CVGS_BENCHMARK

            // Download results
            cvOutput.download(h_cvOutput, cv_stream);
            cvGSOutput.download(h_cvGSOutput, cv_stream);

            cv_stream.waitForCompletion();

            // Verify results
            passed &= compareAndCheck<CV_TYPE_O>(h_cvOutput, h_cvGSOutput);
            if (!passed) {
                std::cout << "Failed for resolution = " << BATCH << "x" << BATCH << std::endl;
            }
        } catch (const cv::Exception& e) {
            if (e.code != -210) {
                error_s << e.what();
                passed = false;
                exception = true;
            }
        } catch (const std::exception& e) {
            error_s << e.what();
            passed = false;
            exception = true;
        }
        if (!passed) {
            if (!exception) {
                std::stringstream ss;
                ss << "benchmark_image_resolution_MAD_loop<" << cvTypeToString<CV_TYPE_I>() << ", " << cvTypeToString<CV_TYPE_O>();
                std::cout << ss.str() << "> failed!! RESULT ERROR: Some results do not match baseline." << std::endl;
            } else {
                std::stringstream ss;
                ss << "benchmark_image_resolution_MAD_loop<" << cvTypeToString<CV_TYPE_I>() << ", " << cvTypeToString<CV_TYPE_O>();
                std::cout << ss.str() << "> failed!! EXCEPTION: " << error_s.str() << std::endl;
            }
        }
    }

    return passed;
}

template <int CV_TYPE_I, int CV_TYPE_O, size_t... Is>
bool launch_benchmark_image_resolution_MAD_loop(std::index_sequence<Is...> seq, cv::cuda::Stream cv_stream, bool enabled) {
    bool passed = true;

    int dummy[] = { (passed &= benchmark_image_resolution_MAD_loop<CV_TYPE_I, CV_TYPE_O, batchValues[Is]>(cv_stream, enabled), 0)... };
    (void)dummy;

    return passed;
}
#endif

int launch() {
#ifdef ENABLE_BENCHMARK
    cv::cuda::Stream cv_stream;

    cv::Mat::setDefaultAllocator(cv::cuda::HostMem::getAllocator(cv::cuda::HostMem::AllocType::PAGE_LOCKED));

    std::unordered_map<std::string, bool> results;
    results["benchmark_image_resolution_MAD_loop"] = true;
    std::make_index_sequence<batchValues.size()> iSeq{};
#define LAUNCH_TESTS(CV_INPUT, CV_OUTPUT) \
    results["benchmark_image_resolution_MAD_loop"] &= launch_benchmark_image_resolution_MAD_loop<CV_INPUT, CV_OUTPUT>(iSeq, cv_stream, true);

    // Warming up for the benchmarks
    warmup = true;
    LAUNCH_TESTS(CV_8UC1, CV_32FC1)
    LAUNCH_TESTS(CV_32FC1, CV_32FC1)
    LAUNCH_TESTS(CV_8UC3, CV_32FC3)
    LAUNCH_TESTS(CV_16UC4, CV_32FC4)
    LAUNCH_TESTS(CV_32SC4, CV_32FC4)
    LAUNCH_TESTS(CV_32FC4, CV_64FC4)
    warmup = false;

    LAUNCH_TESTS(CV_8UC1, CV_32FC1)
    LAUNCH_TESTS(CV_32FC1, CV_32FC1)
    LAUNCH_TESTS(CV_8UC3, CV_32FC3)
    LAUNCH_TESTS(CV_16UC4, CV_32FC4)
    LAUNCH_TESTS(CV_32SC4, CV_32FC4)
    LAUNCH_TESTS(CV_32FC4, CV_64FC4)

    CLOSE_BENCHMARK

    for (const auto& [key, passed] : results) {
        if (passed) {
            std::cout << key << " passed!!" << std::endl;
        } else {
            std::cout << key << " failed!!" << std::endl;
        }
    }

#undef LAUNCH_TESTS
#endif
    return 0;
}