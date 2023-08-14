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

#include <sstream>

#include "testsCommon.cuh"
#include <cvGPUSpeedup.cuh>

#include <opencv2/cudaimgproc.hpp>

template <int I, int OC>
bool test_read_x_write(int NUM_ELEMS_X, int NUM_ELEMS_Y, cv::cuda::Stream& cv_stream, bool enabled) {
    std::stringstream error_s;
    bool passed = true;
    bool exception = false;

    if (enabled) {

        struct Parameters {
            cv::Scalar init;
            cv::Scalar val_sub;
            cv::Scalar val_mul;
            cv::Scalar val_div;
        };

        std::vector<Parameters> params = {
            {{2u}, {0.3f}, {1.f}, {3.2f}},
            {{2u, 37u}, {0.3f, 0.3f}, {1.f, 4.f}, {3.2f, 0.6f}},
            {{2u, 37u, 128u}, {0.3f, 0.3f, 0.3f}, {1.f, 4.f, 3.2f}, {3.2f, 0.6f, 11.8f}},
            {{2u, 37u, 128u, 20u}, {0.3f, 0.3f, 0.3f, 0.3f}, {1.f, 4.f, 3.2f, 0.5f}, {3.2f, 0.6f, 11.8f, 33.f}}
        };

        cv::Scalar val_init = params.at(CV_MAT_CN(OC)-1).init;
        cv::Scalar val_sub = params.at(CV_MAT_CN(OC)-1).val_sub;
        cv::Scalar val_mul = params.at(CV_MAT_CN(OC)-1).val_mul;
        cv::Scalar val_div = params.at(CV_MAT_CN(OC)-1).val_div;
        cv::Scalar val_add = params.at(CV_MAT_CN(OC)-1).val_div;

        try {
            cv::cuda::GpuMat d_input(NUM_ELEMS_Y, NUM_ELEMS_X, I, val_init);
            cv::cuda::GpuMat d_temp(NUM_ELEMS_Y, NUM_ELEMS_X, OC);
            cv::cuda::GpuMat d_output_cv(NUM_ELEMS_Y, NUM_ELEMS_X, OC);
            cv::cuda::GpuMat d_output_cvGS(NUM_ELEMS_Y, NUM_ELEMS_X, OC);

            cv::Mat h_cvResults(NUM_ELEMS_Y, NUM_ELEMS_X, OC);
            cv::Mat h_cvGSResults(NUM_ELEMS_Y, NUM_ELEMS_X, OC);

            // OpenCV version
            d_input.convertTo(d_temp, OC, cv_stream);
            cv::cuda::subtract(d_temp, val_sub, d_output_cv, cv::noArray(), -1, cv_stream);
            cv::cuda::multiply(d_output_cv, val_mul, d_temp, 1.0, -1, cv_stream);
            cv::cuda::divide(d_temp, val_div, d_output_cv, 1.0, -1, cv_stream);
            cv::cuda::add(d_output_cv, val_add, d_output_cv, cv::noArray(), -1, cv_stream);     

            // cvGPUSpeedup version
            cvGS::executeOperations(d_input, d_output_cvGS, cv_stream, 
                                            cvGS::convertTo<I, OC>(),
                                            cvGS::subtract<OC>(val_sub),
                                            cvGS::multiply<OC>(val_mul),
                                            cvGS::divide<OC>(val_div),
                                            cvGS::add<OC>(val_add));

            // Verify results
            d_output_cv.download(h_cvResults, cv_stream);
            d_output_cvGS.download(h_cvGSResults, cv_stream);

            cv_stream.waitForCompletion();

            passed = compareAndCheck<OC>(NUM_ELEMS_X, NUM_ELEMS_Y, h_cvResults, h_cvGSResults);
            
        } catch (const std::exception& e) {
            error_s << e.what();
            passed = false;
            exception = true;
        }

        if (!passed) {
            if (!exception) {
                std::stringstream ss;
                ss << "test_read_x_write<" << cvTypeToString<I>() << ", " << cvTypeToString<OC>();
                std::cout << ss.str() << "> failed!! RESULT ERROR: Some results do not match baseline." << std::endl;
            } else {
                std::stringstream ss;
                ss << "test_read_x_write<" << cvTypeToString<I>() << ", " << cvTypeToString<OC>();
                std::cout << ss.str() << "> failed!! EXCEPTION: " << error_s.str() << std::endl;
            }
        }

    }

    return passed;
}

bool testCvtColor(int NUM_ELEMS_X, int NUM_ELEMS_Y, cv::cuda::Stream& cv_stream, bool enabled) {
    std::stringstream error_s;
    bool passed = true;
    bool exception = false;

    if (enabled) {
        try {
            cv::cuda::GpuMat d_input4(NUM_ELEMS_Y, NUM_ELEMS_X, CV_8UC4, cv::Scalar(1u,2u,3u,4u));
            cv::cuda::GpuMat d_input3(NUM_ELEMS_Y, NUM_ELEMS_X, CV_8UC3, cv::Scalar(1u, 2u, 3u));
            cv::cuda::GpuMat d_output(NUM_ELEMS_Y, NUM_ELEMS_X, CV_8UC3);
            cv::cuda::GpuMat d_cvGSoutput(NUM_ELEMS_Y, NUM_ELEMS_X, CV_8UC3);

            cv::cuda::cvtColor(d_input4, d_output, cv::COLOR_RGBA2BGR, 0, cv_stream);
            cvGS::executeOperations(d_input4, d_cvGSoutput, cv_stream, cvGS::cvtColor<CV_8UC4, cv::COLOR_RGBA2BGR>());

            cv_stream.waitForCompletion();

            cv::Mat h_output(d_output);
            cv::Mat h_cvGSoutput(d_cvGSoutput);

            passed = compareAndCheck<CV_8UC3>(NUM_ELEMS_X, NUM_ELEMS_Y, h_output, h_cvGSoutput);

            cv::cuda::cvtColor(d_input3, d_output, cv::COLOR_RGB2BGR, 0, cv_stream);
            cvGS::executeOperations(d_input3, d_cvGSoutput, cv_stream, cvGS::cvtColor<CV_8UC3, cv::COLOR_RGB2BGR>());
            d_output.download(h_output, cv_stream);
            d_cvGSoutput.download(h_cvGSoutput, cv_stream);

            cv_stream.waitForCompletion();

            passed &= compareAndCheck<CV_8UC3>(NUM_ELEMS_X, NUM_ELEMS_Y, h_output, h_cvGSoutput);

        } catch (const std::exception& e) {
            error_s << e.what();
            passed = false;
            exception = true;
        }

        if (!passed) {
            if (!exception) {
                std::stringstream ss;
                ss << "testCvtColor";
                std::cout << ss.str() << " failed!! RESULT ERROR: Some results do not match baseline." << std::endl;
            }
            else {
                std::stringstream ss;
                ss << "testCvtColor";
                std::cout << ss.str() << "> failed!! EXCEPTION: " << error_s.str() << std::endl;
            }
        }
    }

    return passed;
}

int main() {
    constexpr size_t NUM_ELEMS_X = 3840;
    constexpr size_t NUM_ELEMS_Y = 2160;

    cv::cuda::Stream cv_stream;

    cv::Mat::setDefaultAllocator(cv::cuda::HostMem::getAllocator(cv::cuda::HostMem::AllocType::PAGE_LOCKED));

    std::unordered_map<std::string, bool> results;
    results["test_read_x_write"] = true;
    results["testCvtColor"] = true;

    #define LAUNCH_TESTS(CV_INPUT, CV_OUTPUT) \
    results["test_read_x_write"] &= test_read_x_write<CV_INPUT, CV_OUTPUT>(NUM_ELEMS_X, NUM_ELEMS_Y, cv_stream, true);

    LAUNCH_TESTS(CV_8UC1, CV_32FC1)
    LAUNCH_TESTS(CV_8SC1, CV_32FC1)
    LAUNCH_TESTS(CV_16UC1, CV_32FC1)
    LAUNCH_TESTS(CV_16SC1, CV_32FC1)
    LAUNCH_TESTS(CV_32SC1, CV_32FC1)
    LAUNCH_TESTS(CV_32FC1, CV_32FC1)
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

    #undef LAUNCH_TESTS

    results["testCvtColor"] &= testCvtColor(NUM_ELEMS_X, NUM_ELEMS_Y, cv_stream, true);

    for (const auto& [key, passed] : results) {
        if (passed) {
            std::cout << key << " passed!!" << std::endl;
        } else {
            std::cout << key << " failed!!" << std::endl;
        }
    }

    return 0;
}