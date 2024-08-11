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

#include "tests/testsCommon.cuh"
#include <cvGPUSpeedup.cuh>
#include <opencv2/cudaimgproc.hpp>

#include "tests/main.h"

#ifdef ENABLE_BENCHMARK
constexpr char VARIABLE_DIMENSION[]{ "Number of Operations" };
#ifndef CUDART_MAJOR_VERSION
#error CUDART_MAJOR_VERSION Undefined!
#elif (CUDART_MAJOR_VERSION == 11)
constexpr size_t NUM_EXPERIMENTS = 15;
constexpr size_t FIRST_VALUE = 2;
constexpr size_t INCREMENT = 50;
#elif (CUDART_MAJOR_VERSION == 12)
constexpr size_t NUM_EXPERIMENTS = 60;
constexpr size_t FIRST_VALUE = 2;
constexpr size_t INCREMENT = 100;
#endif // CUDART_MAJOR_VERSION

constexpr std::array<size_t, NUM_EXPERIMENTS> batchValues = arrayIndexSecuence<FIRST_VALUE, INCREMENT, NUM_EXPERIMENTS>;

using namespace fk;

template <int CV_TYPE_I, int CV_TYPE_O, int OPS_PER_ITER, size_t NumOps, typename DeviceFunction>
struct VerticalFusion {
    static inline void execute(const std::array<cv::cuda::GpuMat, 50>& crops,
        const int& BATCH,
        const cv::cuda::Stream& cv_stream,
        const float& alpha,
        const cv::cuda::GpuMat& d_tensor_output,
        const cv::Size& cropSize,
        const DeviceFunction& dFunc) {
        using InputType = CUDA_T(CV_TYPE_I);
        using OutputType = CUDA_T(CV_TYPE_O);
        using Loop = Binary<StaticLoop<StaticLoop<
            typename DeviceFunction::Operation, INCREMENT / OPS_PER_ITER>, NumOps / INCREMENT>>;

        Loop loop;
        loop.params = dFunc.params;

        cvGS::executeOperations(crops, BATCH, cv_stream, cvGS::convertTo<CV_TYPE_I, CV_TYPE_O>((float)alpha), loop, cvGS::write<CV_TYPE_O>(d_tensor_output, cropSize));
    }
};

template <int CV_TYPE_I, int CV_TYPE_O, size_t BATCH>
bool benchmark_vertical_fusion_loopDivide(size_t NUM_ELEMS_X, size_t NUM_ELEMS_Y, cv::cuda::Stream& cv_stream, bool enabled) {
    constexpr size_t REAL_BATCH{ 50 };
    std::stringstream error_s;
    bool passed = true;
    bool exception = false;

    if (enabled) {
        struct Parameters {
            const cv::Scalar init;
            const cv::Scalar val_mul;
        };

        double alpha = 1.0;

        const Parameters one{ {30u}, {0.54f} };

        const cv::Scalar val_init = one.init;
        const cv::Scalar val_mul = one.val_mul;
        try {
            const cv::Size cropSize(NUM_ELEMS_X, NUM_ELEMS_Y);

            std::array<cv::cuda::GpuMat, REAL_BATCH> crops;
            std::array<cv::cuda::GpuMat, REAL_BATCH> d_output_cv;
            std::array<cv::Mat, REAL_BATCH>          h_output_cv;
            for (int crop_i = 0; crop_i < REAL_BATCH; crop_i++) {
                crops[crop_i] = cv::cuda::GpuMat(cropSize, CV_TYPE_I, val_init);
                h_output_cv[crop_i].create(cropSize, CV_TYPE_O);
                d_output_cv[crop_i].create(cropSize, CV_TYPE_O);
            }

            cv::cuda::GpuMat d_output_cvGS(REAL_BATCH, cropSize.width * cropSize.height, CV_TYPE_O);
            d_output_cvGS.step = cropSize.width * cropSize.height * sizeof(CUDA_T(CV_TYPE_O));
            cv::Mat h_output_cvGS(REAL_BATCH, cropSize.width * cropSize.height, CV_TYPE_O);

            START_OCV_BENCHMARK
            // OpenCV version
            constexpr int OPS_PER_ITERATION = 2;

            for (int crop_i = 0; crop_i < REAL_BATCH; crop_i++) {
                crops[crop_i].convertTo(d_output_cv[crop_i], CV_TYPE_O, alpha, cv_stream);
                for (int numOp = 0; numOp < BATCH; numOp+= OPS_PER_ITERATION) {
                    cv::cuda::divide(d_output_cv[crop_i], val_mul, d_output_cv[crop_i], 1.0, -1, cv_stream);
                    cv::cuda::divide(d_output_cv[crop_i], val_mul, d_output_cv[crop_i], 1.0, -1, cv_stream);
                }
            }

            STOP_OCV_START_CVGS_BENCHMARK
            using InputType = CUDA_T(CV_TYPE_I);
            using OutputType = CUDA_T(CV_TYPE_O);

            const OutputType val{ cvGS::cvScalar2CUDAV<CV_TYPE_O>::get(val_mul) };

            // cvGPUSpeedup
            using DeviceFunction = Binary<Div<OutputType>, Div<OutputType>>;
            const DeviceFunction dFunc(val, val);
            VerticalFusion<CV_TYPE_I, CV_TYPE_O, OPS_PER_ITERATION, BATCH, DeviceFunction>::execute(crops, REAL_BATCH, cv_stream, alpha, d_output_cvGS, cropSize, dFunc);
            
            STOP_CVGS_BENCHMARK

            // Download results
            for (int crop_i = 0; crop_i < REAL_BATCH; crop_i++) {
                d_output_cv[crop_i].download(h_output_cv[crop_i], cv_stream);
            }
            d_output_cvGS.download(h_output_cvGS, cv_stream);

            cv_stream.waitForCompletion();

            // Verify results
            for (int crop_i = 0; crop_i < REAL_BATCH; crop_i++) {
                cv::Mat cvRes = h_output_cv[crop_i];
                cv::Mat cvGSRes = cv::Mat(cropSize.height, cropSize.width, CV_TYPE_O, h_output_cvGS.row(crop_i).data);
                bool passedThisTime = compareAndCheck<CV_TYPE_O>(cropSize.width, cropSize.height, cvRes, cvGSRes);
                passed &= passedThisTime;
                if (!passedThisTime) {
                    int a = 0;
                    a++;
                }
            }
            if (!passed) {
                std::cout << "Failed for num fused operations = " << BATCH << std::endl;
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
                ss << "benchmark_vertical_fusion_loopDivide<" << cvTypeToString<CV_TYPE_I>() << ", " << cvTypeToString<CV_TYPE_O>();
                std::cout << ss.str() << "> failed!! RESULT ERROR: Some results do not match baseline." << std::endl;
            } else {
                std::stringstream ss;
                ss << "benchmark_vertical_fusion_loopDivide<" << cvTypeToString<CV_TYPE_I>() << ", " << cvTypeToString<CV_TYPE_O>();
                std::cout << ss.str() << "> failed!! EXCEPTION: " << error_s.str() << std::endl;
            }
        }
    }

    return passed;
}

template <int CV_TYPE_I, int CV_TYPE_O, size_t... Is>
bool launch_benchmark_vertical_fusion_loopDivide(const size_t NUM_ELEMS_X, const size_t NUM_ELEMS_Y, std::index_sequence<Is...> seq, cv::cuda::Stream cv_stream, bool enabled) {
    bool passed = true;

    int dummy[] = { (passed &= benchmark_vertical_fusion_loopDivide<CV_TYPE_I, CV_TYPE_O, batchValues[Is]>(NUM_ELEMS_X, NUM_ELEMS_Y, cv_stream, enabled), 0)... };
    (void)dummy;

    return passed;
}
#endif

int launch() {
#ifdef ENABLE_BENCHMARK
    constexpr size_t NUM_ELEMS_X = 60;
    constexpr size_t NUM_ELEMS_Y = 120;

    cv::cuda::Stream cv_stream;

    cv::Mat::setDefaultAllocator(cv::cuda::HostMem::getAllocator(cv::cuda::HostMem::AllocType::PAGE_LOCKED));

    std::unordered_map<std::string, bool> results;
    results["benchmark_vertical_fusion_loopDivide"] = true;
    std::make_index_sequence<batchValues.size()> iSeq{};
#define LAUNCH_TESTS(CV_INPUT, CV_OUTPUT) \
    results["benchmark_vertical_fusion_loopDivide"] &= launch_benchmark_vertical_fusion_loopDivide<CV_INPUT, CV_OUTPUT>(NUM_ELEMS_X, NUM_ELEMS_Y, iSeq, cv_stream, true);

    LAUNCH_TESTS(CV_8UC1, CV_32FC1)

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