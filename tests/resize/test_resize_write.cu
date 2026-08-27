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

// Tests the cvGS::resize overload that computes the destination size from the
// fx and fy scale factors, instead of using a cv::Size.
template <int I, int O>
bool test_resize_write_scale_factors(int NUM_ELEMS_X, int NUM_ELEMS_Y, cv::cuda::Stream& cv_stream, bool enabled) {
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

            constexpr double fxUp = 1.5;
            constexpr double fyUp = 1.5;
            constexpr double fxDown = 0.25;
            constexpr double fyDown = 0.25;

            const cv::Size up(static_cast<int>(NUM_ELEMS_X * fxUp), static_cast<int>(NUM_ELEMS_Y * fyUp));
            const cv::Size down(static_cast<int>(NUM_ELEMS_X * fxDown), static_cast<int>(NUM_ELEMS_Y * fyDown));

            cv::cuda::GpuMat d_down(down, I);
            cv::cuda::GpuMat d_up(up, I);

            cv::cuda::GpuMat d_down_cvGS(down, I);
            cv::cuda::GpuMat d_up_cvGS(up, I);

            // Execute cvGS first to avoid OpenCV exceptions
            cvGS::executeOperations(cv_stream, cvGS::resize<I, cv::INTER_LINEAR>(d_input, cv::Size(0, 0), fxUp, fyUp),
                                               cvGS::convertTo<CV_MAKETYPE(CV_32F, CV_MAT_CN(I)), I>(),
                                               cvGS::write<I>(d_up_cvGS));
            cvGS::executeOperations(cv_stream, cvGS::resize<I, cv::INTER_LINEAR>(d_input, cv::Size(0, 0), fxDown, fyDown),
                                               cvGS::convertTo<CV_MAKETYPE(CV_32F, CV_MAT_CN(I)), I>(),
                                               cvGS::write<I>(d_down_cvGS));

            cv::cuda::resize(d_input, d_up, cv::Size(), fxUp, fyUp, cv::INTER_LINEAR, cv_stream);
            cv::cuda::resize(d_input, d_down, cv::Size(), fxDown, fyDown, cv::INTER_LINEAR, cv_stream);

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
                ss << "test_resize_write_scale_factors<" << cvTypeToString<I>() << ", " << cvTypeToString<O>();
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
                ss << "test_resize_write_scale_factors<" << cvTypeToString<I>() << ", " << cvTypeToString<O>();
                std::cout << ss.str() << "> failed!! RESULT ERROR: Some results do not match baseline." << std::endl;
            } else {
                std::stringstream ss;
                ss << "test_resize_write_scale_factors<" << cvTypeToString<I>() << ", " << cvTypeToString<O>();
                std::cout << ss.str() << "> failed!! EXCEPTION: " << error_s.str() << std::endl;
            }
        }
    }

    return passed;
}

// Tests the cvGS::resize overload that only receives the destination size, and
// therefore has to be fused with a previous read Operation, by using then().
template <int I, int O>
bool test_resize_write_fused(int NUM_ELEMS_X, int NUM_ELEMS_Y, cv::cuda::Stream& cv_stream, bool enabled) {
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

            const cv::Size up(3870, 2260); // x,y
            const cv::Size down(300, 500); // x,y

            cv::cuda::GpuMat d_down(down, I);
            cv::cuda::GpuMat d_up(up, I);

            cv::cuda::GpuMat d_down_cvGS(down, I);
            cv::cuda::GpuMat d_up_cvGS(up, I);

            const auto readOp =
                fk::PerThreadRead<fk::ND::_2D, CUDA_T(I)>::build(cvGS::gpuMat2Ptr2D<CUDA_T(I)>(d_input).ptr());

            // Execute cvGS first to avoid OpenCV exceptions
            cvGS::executeOperations(cv_stream, readOp.then(cvGS::resize<cv::INTER_LINEAR>(up)),
                                               cvGS::convertTo<CV_MAKETYPE(CV_32F, CV_MAT_CN(I)), I>(),
                                               cvGS::write<I>(d_up_cvGS));
            cvGS::executeOperations(cv_stream, readOp.then(cvGS::resize<cv::INTER_LINEAR>(down)),
                                               cvGS::convertTo<CV_MAKETYPE(CV_32F, CV_MAT_CN(I)), I>(),
                                               cvGS::write<I>(d_down_cvGS));

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
                ss << "test_resize_write_fused<" << cvTypeToString<I>() << ", " << cvTypeToString<O>();
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
                ss << "test_resize_write_fused<" << cvTypeToString<I>() << ", " << cvTypeToString<O>();
                std::cout << ss.str() << "> failed!! RESULT ERROR: Some results do not match baseline." << std::endl;
            } else {
                std::stringstream ss;
                ss << "test_resize_write_fused<" << cvTypeToString<I>() << ", " << cvTypeToString<O>();
                std::cout << ss.str() << "> failed!! EXCEPTION: " << error_s.str() << std::endl;
            }
        }
    }

    return passed;
}

// Tests the cvGS::resize overload that preserves the aspect ratio, filling the
// rest of the destination image with a background value.
template <int I, int O>
bool test_resize_write_aspect_ratio(int NUM_ELEMS_X, int NUM_ELEMS_Y, cv::cuda::Stream& cv_stream, bool enabled) {
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
        const cv::Scalar background = cvGS::cvScalar_set<CV_MAKETYPE(CV_32F, CV_MAT_CN(I))>(128.f);

        try {
            // We use a source with an aspect ratio different than the destination one
            cv::cuda::GpuMat d_input(NUM_ELEMS_Y, NUM_ELEMS_X, I, val_init);

            const cv::Size dstSize(512, 512); // x,y

            // Compute the size that preserves the aspect ratio, the same way cvGS does
            float scaleFactor = dstSize.height / static_cast<float>(NUM_ELEMS_Y);
            int targetWidth = static_cast<int>(std::round(scaleFactor * NUM_ELEMS_X));
            int targetHeight = dstSize.height;
            if (targetWidth > dstSize.width) {
                scaleFactor = dstSize.width / static_cast<float>(NUM_ELEMS_X);
                targetWidth = dstSize.width;
                targetHeight = static_cast<int>(std::round(scaleFactor * NUM_ELEMS_Y));
            }
            const cv::Size targetSize(targetWidth, targetHeight);

            cv::cuda::GpuMat d_output(dstSize, I, background);
            cv::cuda::GpuMat d_output_cvGS(dstSize, I);

            const auto readOp =
                fk::PerThreadRead<fk::ND::_2D, CUDA_T(I)>::build(cvGS::gpuMat2Ptr2D<CUDA_T(I)>(d_input).ptr());
            constexpr int DEFAULT_TYPE = CV_MAKETYPE(CV_32F, CV_MAT_CN(I));

            // Execute cvGS first to avoid OpenCV exceptions
            cvGS::executeOperations(cv_stream,
                readOp.then(cvGS::resize<cv::INTER_LINEAR, fk::AspectRatio::PRESERVE_AR, DEFAULT_TYPE>(dstSize, background)),
                cvGS::convertTo<DEFAULT_TYPE, I>(),
                cvGS::write<I>(d_output_cvGS));

            // OpenCV version: resize into the centered region of interest
            const int xOffset = (dstSize.width - targetSize.width) / 2;
            const int yOffset = (dstSize.height - targetSize.height) / 2;
            cv::cuda::GpuMat d_roi = d_output(cv::Rect(xOffset, yOffset, targetSize.width, targetSize.height));
            cv::cuda::resize(d_input, d_roi, targetSize, 0., 0., cv::INTER_LINEAR, cv_stream);

            cv::Mat h_output, h_output_cvGS;
            d_output.download(h_output, cv_stream);
            d_output_cvGS.download(h_output_cvGS, cv_stream);

            cv_stream.waitForCompletion();

            passed &= compareAndCheck<I>(dstSize.width, dstSize.height, h_output, h_output_cvGS);

        } catch (const cv::Exception& e) {
            if (e.code != -210) {
                error_s << e.what();
                passed = false;
                exception = true;
            } else {
                std::stringstream ss;
                ss << "test_resize_write_aspect_ratio<" << cvTypeToString<I>() << ", " << cvTypeToString<O>();
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
                ss << "test_resize_write_aspect_ratio<" << cvTypeToString<I>() << ", " << cvTypeToString<O>();
                std::cout << ss.str() << "> failed!! RESULT ERROR: Some results do not match baseline." << std::endl;
            } else {
                std::stringstream ss;
                ss << "test_resize_write_aspect_ratio<" << cvTypeToString<I>() << ", " << cvTypeToString<O>();
                std::cout << ss.str() << "> failed!! EXCEPTION: " << error_s.str() << std::endl;
            }
        }
    }

    return passed;
}

// Tests the cvGS::resize overload that resizes a batch of images at once.
template <int I, int O, size_t BATCH>
bool test_resize_write_batch(int NUM_ELEMS_X, int NUM_ELEMS_Y, cv::cuda::Stream& cv_stream, bool enabled) {
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
            const cv::Size down(300, 500); // x,y

            std::array<cv::cuda::GpuMat, BATCH> d_inputs;
            std::array<cv::cuda::GpuMat, BATCH> d_outputs_cv;
            for (size_t i = 0; i < BATCH; i++) {
                d_inputs[i] = cv::cuda::GpuMat(NUM_ELEMS_Y, NUM_ELEMS_X, I, val_init);
                d_outputs_cv[i] = cv::cuda::GpuMat(down, I);
            }

            cv::cuda::GpuMat d_tensor_output(BATCH, down.width * down.height * CV_MAT_CN(I), CV_MAT_DEPTH(I));
            d_tensor_output.step = down.width * down.height * CV_MAT_CN(I) * sizeof(BASE_CUDA_T(I));

            // Execute cvGS first to avoid OpenCV exceptions
            cvGS::executeOperations(cv_stream,
                cvGS::resize<I, cv::INTER_LINEAR, BATCH>(d_inputs, down, BATCH),
                cvGS::convertTo<CV_MAKETYPE(CV_32F, CV_MAT_CN(I)), I>(),
                cvGS::write<I>(d_tensor_output, down));

            for (size_t i = 0; i < BATCH; i++) {
                cv::cuda::resize(d_inputs[i], d_outputs_cv[i], down, 0., 0., cv::INTER_LINEAR, cv_stream);
            }

            cv::Mat h_tensor_output(BATCH, down.width * down.height * CV_MAT_CN(I), CV_MAT_DEPTH(I));
            d_tensor_output.download(h_tensor_output, cv_stream);

            std::array<cv::Mat, BATCH> h_outputs_cv;
            for (size_t i = 0; i < BATCH; i++) {
                d_outputs_cv[i].download(h_outputs_cv[i], cv_stream);
            }

            cv_stream.waitForCompletion();

            for (size_t i = 0; i < BATCH; i++) {
                cv::Mat row = h_tensor_output.row(static_cast<int>(i));
                cv::Mat h_cvGSResult(down.height, down.width, I, row.data);
                passed &= compareAndCheck<I>(down.width, down.height, h_outputs_cv[i], h_cvGSResult);
            }

        } catch (const cv::Exception& e) {
            if (e.code != -210) {
                error_s << e.what();
                passed = false;
                exception = true;
            } else {
                std::stringstream ss;
                ss << "test_resize_write_batch<" << cvTypeToString<I>() << ", " << cvTypeToString<O>();
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
                ss << "test_resize_write_batch<" << cvTypeToString<I>() << ", " << cvTypeToString<O>();
                std::cout << ss.str() << "> failed!! RESULT ERROR: Some results do not match baseline." << std::endl;
            } else {
                std::stringstream ss;
                ss << "test_resize_write_batch<" << cvTypeToString<I>() << ", " << cvTypeToString<O>();
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
    results["test_resize_write_scale_factors"] = true;
    results["test_resize_write_fused"] = true;
    results["test_resize_write_aspect_ratio"] = true;
    results["test_resize_write_batch"] = true;

    #define LAUNCH_TESTS(CV_INPUT, CV_OUTPUT) \
    results["test_resize_write"] &= test_resize_write<CV_INPUT, CV_OUTPUT>(NUM_ELEMS_X, NUM_ELEMS_Y, cv_stream, true); \
    results["test_resize_write_scale_factors"] &= test_resize_write_scale_factors<CV_INPUT, CV_OUTPUT>(NUM_ELEMS_X, NUM_ELEMS_Y, cv_stream, true); \
    results["test_resize_write_fused"] &= test_resize_write_fused<CV_INPUT, CV_OUTPUT>(NUM_ELEMS_X, NUM_ELEMS_Y, cv_stream, true); \
    results["test_resize_write_aspect_ratio"] &= test_resize_write_aspect_ratio<CV_INPUT, CV_OUTPUT>(NUM_ELEMS_X, NUM_ELEMS_Y, cv_stream, true); \
    results["test_resize_write_batch"] &= test_resize_write_batch<CV_INPUT, CV_OUTPUT, 3>(NUM_ELEMS_X, NUM_ELEMS_Y, cv_stream, true);

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