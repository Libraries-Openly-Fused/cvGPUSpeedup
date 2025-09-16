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

#include "tests/testsCommon.cuh"
#include <cvGPUSpeedup.cuh>
#include <opencv2/cudaimgproc.hpp>

#include "tests/main.h"

#ifdef ENABLE_BENCHMARK
constexpr char VARIABLE_DIMENSION[] {"Batch size"};
#define BENCHMARK_ENABLED true
#else
#define BENCHMARK_ENABLED false
#endif

constexpr size_t FIRST_VALUE = 50;
constexpr size_t INCREMENT = 1;
constexpr size_t NUM_EXPERIMENTS = 1;
constexpr std::array<size_t, NUM_EXPERIMENTS> batchValues = arrayIndexSecuence<FIRST_VALUE, INCREMENT, NUM_EXPERIMENTS>;

template <int CV_TYPE_I, int CV_TYPE_O, int BATCH>
bool test_batchread_x_write3D(size_t NUM_ELEMS_X, size_t NUM_ELEMS_Y, cv::cuda::Stream& cv_stream, bool enabled) {
    std::stringstream error_s;
    bool passed = true;
    bool exception = false;
    
    if (enabled) {
        struct Parameters {
            const cv::Scalar init;
            const cv::Scalar alpha;
            const cv::Scalar val_sub;
            const cv::Scalar val_div;
        };

        double alpha = 1.0;

        const Parameters one{ {10u}, {alpha}, {1.5f}, {3.2f} };
        const Parameters two{ {10u, 20u}, {alpha, alpha}, {1.5f, 4.f}, {3.2f, 0.6f} };
        const Parameters three{ {10u, 20u, 30u}, {alpha, alpha, alpha}, {1.5f, 4.f, 3.2f}, {3.2f, 0.6f, 11.8f} };
        const Parameters four{ {10u, 20u, 30u, 40u}, {alpha, alpha, alpha, alpha}, {1.5f, 4.f, 3.2f, 0.5f}, {3.2f, 0.6f, 11.8f, 33.f} };
        const std::array<Parameters, 4> params{ one, two, three, four };

        const cv::Scalar val_init = params.at(CV_MAT_CN(CV_TYPE_O) - 1).init;
        const cv::Scalar val_alpha = params.at(CV_MAT_CN(CV_TYPE_O) - 1).alpha;
        const cv::Scalar val_sub = params.at(CV_MAT_CN(CV_TYPE_O) - 1).val_sub;
        const cv::Scalar val_div = params.at(CV_MAT_CN(CV_TYPE_O) - 1).val_div;
        try {
            const cv::Size cropSize(60, 120);
            cv::cuda::GpuMat d_input((int)NUM_ELEMS_Y, (int)NUM_ELEMS_X, CV_TYPE_I, val_init);
            std::array<cv::cuda::GpuMat, BATCH> d_output_cv;
            std::array<cv::Mat, BATCH> h_cvResults;
            std::array<cv::Mat, BATCH> h_cvGSResults;

            cv::cuda::GpuMat d_temp(cropSize, CV_TYPE_O);
            cv::cuda::GpuMat d_temp2(cropSize, CV_TYPE_O);

            cv::cuda::GpuMat d_tensor_output(BATCH,
                cropSize.width * cropSize.height,
                CV_TYPE_O);
            d_tensor_output.step = cropSize.width * cropSize.height * sizeof(CUDA_T(CV_TYPE_O));

            cv::Mat diff(cropSize, CV_TYPE_O);
            cv::Mat h_tensor_output(BATCH, cropSize.width * cropSize.height, CV_TYPE_I);

            std::array<cv::cuda::GpuMat, BATCH> crops;
            for (int crop_i = 0; crop_i < BATCH; crop_i++) {
                crops[crop_i] = cv::cuda::GpuMat(cropSize, CV_TYPE_I, val_init);
                d_output_cv[crop_i].create(cropSize, CV_TYPE_O);
                h_cvResults[crop_i].create(cropSize, CV_TYPE_O);
            }
            START_OCV_BENCHMARK
            // OpenCV version
            for (int crop_i = 0; crop_i < BATCH; crop_i++) {
                crops[crop_i].convertTo(d_temp, CV_TYPE_O, alpha, cv_stream);
                cv::cuda::subtract(d_temp, val_sub, d_temp2, cv::noArray(), -1, cv_stream);
                cv::cuda::divide(d_temp2, val_div, d_output_cv[crop_i], 1.0, -1, cv_stream);
            }

            STOP_OCV_START_CVGS_BENCHMARK
            cvGS::executeOperations<false>(crops, cv_stream,
                                            cvGS::convertTo<CV_TYPE_I, CV_TYPE_O>((float)alpha),
                                            cvGS::subtract<CV_TYPE_O>(val_sub),
                                            cvGS::divide<CV_TYPE_O>(val_div),
                                            cvGS::write<CV_TYPE_O>(d_tensor_output, cropSize));
            STOP_CVGS_BENCHMARK

            d_tensor_output.download(h_tensor_output, cv_stream);

            // Verify results
            for (int crop_i = 0; crop_i < BATCH; crop_i++) {
                d_output_cv[crop_i].download(h_cvResults[crop_i], cv_stream);
            }

            cv_stream.waitForCompletion();

            for (int crop_i = 0; crop_i < BATCH; crop_i++) {
                cv::Mat cvRes = h_cvResults[crop_i];
                cv::Mat cvGSRes = cv::Mat(cropSize.height, cropSize.width, CV_TYPE_O, h_tensor_output.row(crop_i).data);
                bool passedThisTime = compareAndCheck<CV_TYPE_O>(cropSize.width, cropSize.height, cvRes, cvGSRes);
                if (!passedThisTime) { std::cout << "Failed on crop idx=" << crop_i << std::endl; }
                passed &= passedThisTime;
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
                ss << "test_batchread_x_write3D<" << cvTypeToString<CV_TYPE_I>() << ", " << cvTypeToString<CV_TYPE_O>();
                std::cout << ss.str() << "> failed!! RESULT ERROR: Some results do not match baseline." << std::endl;
            } else {
                std::stringstream ss;
                ss << "test_batchread_x_write3D<" << cvTypeToString<CV_TYPE_I>() << ", " << cvTypeToString<CV_TYPE_O>();
                std::cout << ss.str() << "> failed!! EXCEPTION: " << error_s.str() << std::endl;
            }
        }
    }

    return passed;
}

template <int CV_TYPE_I, int CV_TYPE_O, size_t... Is>
bool launch_test_batchread_x_write3D(const size_t NUM_ELEMS_X, const size_t NUM_ELEMS_Y, std::index_sequence<Is...> seq, cv::cuda::Stream cv_stream, bool enabled) {
    bool passed = true;

    int dummy[] = {(passed &= test_batchread_x_write3D<CV_TYPE_I, CV_TYPE_O, batchValues[Is]>(NUM_ELEMS_X, NUM_ELEMS_Y, cv_stream, enabled), 0)...};
    (void)dummy;

    return passed;
}

int launch() {
    constexpr size_t NUM_ELEMS_X = 3840;
    constexpr size_t NUM_ELEMS_Y = 2160;

    cv::cuda::Stream cv_stream;

    cv::Mat::setDefaultAllocator(cv::cuda::HostMem::getAllocator(cv::cuda::HostMem::AllocType::PAGE_LOCKED));

    std::unordered_map<std::string, bool> results;
    results["test_batchread_x_write3D"] = true;
    std::make_index_sequence<batchValues.size()> iSeq {};

#define LAUNCH_TESTS(CV_INPUT, CV_OUTPUT) \
    results["test_batchread_x_write3D"] &= launch_test_batchread_x_write3D<CV_INPUT, CV_OUTPUT>(NUM_ELEMS_X, NUM_ELEMS_Y, iSeq, cv_stream, true);
    
#ifdef ENABLE_BENCHMARK
    
    // Warming up for the benchmarks
    warmup = true;
    LAUNCH_TESTS(CV_8UC1, CV_32FC1)
    LAUNCH_TESTS(CV_8SC1, CV_32FC1)
    LAUNCH_TESTS(CV_16UC1, CV_32FC1)
    LAUNCH_TESTS(CV_16SC1, CV_32FC1)
    LAUNCH_TESTS(CV_32SC1, CV_32FC1)
    LAUNCH_TESTS(CV_32FC1, CV_32FC1)
    LAUNCH_TESTS(CV_32FC1, CV_64FC1)
    LAUNCH_TESTS(CV_64FC1, CV_64FC1)
    LAUNCH_TESTS(CV_8UC2, CV_32FC2)
    LAUNCH_TESTS(CV_8UC3, CV_32FC3)
    LAUNCH_TESTS(CV_8UC4, CV_32FC4)
    LAUNCH_TESTS(CV_8SC2, CV_32FC2)
    LAUNCH_TESTS(CV_8SC3, CV_32FC3)
    LAUNCH_TESTS(CV_8SC4, CV_32FC4)
    LAUNCH_TESTS(CV_16UC2, CV_32FC2)
    LAUNCH_TESTS(CV_16UC3, CV_32FC3)
    LAUNCH_TESTS(CV_16UC4, CV_32FC4)
    LAUNCH_TESTS(CV_16SC2, CV_32FC2)
    LAUNCH_TESTS(CV_16SC3, CV_32FC3)
    LAUNCH_TESTS(CV_16SC4, CV_32FC4)
    LAUNCH_TESTS(CV_32SC2, CV_32FC2)
    LAUNCH_TESTS(CV_32SC3, CV_32FC3)
    LAUNCH_TESTS(CV_32SC4, CV_32FC4)
    LAUNCH_TESTS(CV_32FC2, CV_64FC2)
    LAUNCH_TESTS(CV_32FC3, CV_64FC3)
    LAUNCH_TESTS(CV_32FC4, CV_64FC4)
    warmup = false;

#endif

    // Execute experiments
    LAUNCH_TESTS(CV_8UC1, CV_32FC1)
    LAUNCH_TESTS(CV_8SC1, CV_32FC1)
    LAUNCH_TESTS(CV_16UC1, CV_32FC1)
    LAUNCH_TESTS(CV_16SC1, CV_32FC1)
    LAUNCH_TESTS(CV_32SC1, CV_32FC1)
    LAUNCH_TESTS(CV_32FC1, CV_32FC1)
    LAUNCH_TESTS(CV_32FC1, CV_64FC1)
    LAUNCH_TESTS(CV_64FC1, CV_64FC1)
    LAUNCH_TESTS(CV_8UC2, CV_32FC2)
    LAUNCH_TESTS(CV_8UC3, CV_32FC3)
    LAUNCH_TESTS(CV_8UC4, CV_32FC4)
    LAUNCH_TESTS(CV_8SC2, CV_32FC2)
    LAUNCH_TESTS(CV_8SC3, CV_32FC3)
    LAUNCH_TESTS(CV_8SC4, CV_32FC4)
    LAUNCH_TESTS(CV_16UC2, CV_32FC2)
    LAUNCH_TESTS(CV_16UC3, CV_32FC3)
    LAUNCH_TESTS(CV_16UC4, CV_32FC4)
    LAUNCH_TESTS(CV_16SC2, CV_32FC2)
    LAUNCH_TESTS(CV_16SC3, CV_32FC3)
    LAUNCH_TESTS(CV_16SC4, CV_32FC4)
    LAUNCH_TESTS(CV_32SC2, CV_32FC2)
    LAUNCH_TESTS(CV_32SC3, CV_32FC3)
    LAUNCH_TESTS(CV_32SC4, CV_32FC4)
    LAUNCH_TESTS(CV_32FC2, CV_64FC2)
    LAUNCH_TESTS(CV_32FC3, CV_64FC3)
    LAUNCH_TESTS(CV_32FC4, CV_64FC4)

    CLOSE_BENCHMARK

    int returnValue = 0;
    for (const auto& [key, passed] : results) {
        if (passed) {
            std::cout << key << " passed!!" << std::endl;
        } else {
            std::cout << key << " failed!!" << std::endl;
            returnValue = -1;
        }
    }
#undef LAUNCH_TESTS
    return returnValue;
}