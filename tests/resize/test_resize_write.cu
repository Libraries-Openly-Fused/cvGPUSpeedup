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

#include "tests/main.h"

#include "tests/testsCommon.cuh"
#include <cvGPUSpeedup.cuh>

#include <iomanip>

template <int I, int O>
bool test_resize_write(int NUM_ELEMS_X, int NUM_ELEMS_Y, cv::cuda::Stream& cv_stream, bool enabled) {
    std::stringstream error_s;
    bool passed = true;
    bool exception = false;

    if (enabled) {

        struct Parameters {
            cv::Scalar init;
        };

        std::vector<Parameters> params = {
            {{2u}},
            {{2u, 37u}},
            {{2u, 37u, 128u}},
            {{2u, 37u, 128u, 20u}}
        };

        cv::Scalar val_init = params.at(CV_MAT_CN(I)-1).init;

        try {

            cv::cuda::GpuMat d_input(NUM_ELEMS_Y, NUM_ELEMS_X, I, val_init);

            cv::Size up(3870, 2260); // x,y
            cv::Size down(300, 500); // x,y

            cv::cuda::GpuMat d_down(down, I);
            cv::cuda::GpuMat d_up(up, I);

            cv::cuda::GpuMat d_down_cvGS(down, I);
            cv::cuda::GpuMat d_up_cvGS(up, I);

            // Execute cvGS first to avoid OpenCV exceptions
            cvGS::executeOperations(cv_stream, cvGS::resize<I, cv::INTER_LINEAR>(d_input, up, 0., 0.),
                                               cvGS::convertTo<CV_MAKETYPE(CV_32F, CV_MAT_CN(I)), I>(),
                                               cvGS::write<I>(d_up_cvGS));
            cvGS::executeOperations(cv_stream, cvGS::resize<I, cv::INTER_LINEAR>(d_input, down, 0., 0.), cvGS::convertTo<CV_MAKETYPE(CV_32F, CV_MAT_CN(I)), I>(), cvGS::write<I>(d_down_cvGS));

            cv::cuda::resize(d_input, d_up, up, 0., 0., cv::INTER_LINEAR, cv_stream);
            cv::cuda::resize(d_input, d_down, down, 0., 0., cv::INTER_LINEAR, cv_stream);

            cv::Mat h_up, h_up_cvGS;
            cv::Mat h_down, h_down_cvGS;

            d_up.download(h_up, cv_stream);
            d_up_cvGS.download(h_up_cvGS, cv_stream);
            d_down.download(h_down, cv_stream);
            d_down_cvGS.download(h_down_cvGS, cv_stream);

            cv_stream.waitForCompletion();

            passed &= compareAndCheck<I>(up.width, up.height, h_up, h_up_cvGS);
            passed &= compareAndCheck<I>(down.width, down.height, h_down, h_down_cvGS);

        } catch (const cv::Exception& e) {
            if (e.code != -210) {
                error_s << e.what();
                passed = false;
                exception = true;
            } else {
                std::stringstream ss;
                ss << "test_resize_write<" << cvTypeToString<I>() << ", " << cvTypeToString<O>();
                std::cout << ss.str() << "> not supported by OpenCV" << std::endl;
            }
        } catch (const std::exception& e) {
            error_s << e.what();
            passed = false;
            exception = true;
        } 

        if (!passed) {
            if (!exception) {
                std::stringstream ss;
                ss << "test_resize_write<" << cvTypeToString<I>() << ", " << cvTypeToString<O>();
                std::cout << ss.str() << "> failed!! RESULT ERROR: Some results do not match baseline." << std::endl;
            } else {
                std::stringstream ss;
                ss << "test_resize_write<" << cvTypeToString<I>() << ", " << cvTypeToString<O>();
                std::cout << ss.str() << "> failed!! EXCEPTION: " << error_s.str() << std::endl;
            }
        }
    }

    return passed;
}

int launch() {
    constexpr size_t NUM_ELEMS_X = 3840;
    constexpr size_t NUM_ELEMS_Y = 2160;

    cv::cuda::Stream cv_stream;

    cv::Mat::setDefaultAllocator(cv::cuda::HostMem::getAllocator(cv::cuda::HostMem::AllocType::PAGE_LOCKED));

    std::unordered_map<std::string, bool> results;
    results["test_resize_write"] = true;

    #define LAUNCH_TESTS(CV_INPUT, CV_OUTPUT) \
    results["test_resize_write"] &= test_resize_write<CV_INPUT, CV_OUTPUT>(NUM_ELEMS_X, NUM_ELEMS_Y, cv_stream, true);

    LAUNCH_TESTS(CV_8UC1, CV_32FC1)
    LAUNCH_TESTS(CV_16UC1, CV_32FC1)
    LAUNCH_TESTS(CV_16SC1, CV_32FC1)
    LAUNCH_TESTS(CV_32FC1, CV_32FC1)
    LAUNCH_TESTS(CV_8UC3, CV_32FC3)
    LAUNCH_TESTS(CV_8UC4, CV_32FC4)
    LAUNCH_TESTS(CV_16UC3, CV_32FC3)
    LAUNCH_TESTS(CV_16UC4, CV_32FC4)
    LAUNCH_TESTS(CV_16SC3, CV_32FC3)
    LAUNCH_TESTS(CV_16SC4, CV_32FC4)
    LAUNCH_TESTS(CV_32FC3, CV_64FC3)
    LAUNCH_TESTS(CV_32FC4, CV_64FC4)

#undef LAUNCH_TESTS

    int returnValue = 0;
    for (const auto& [key, passed] : results) {
        if (passed) {
            std::cout << key << " passed!!" << std::endl;
        } else {
            std::cout << key << " failed!!" << std::endl;
            returnValue = -1;
        }
    }

    return returnValue;
}