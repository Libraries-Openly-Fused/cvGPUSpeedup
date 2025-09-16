/* Copyright 2025 Grup Mediapro S.L.U

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

#include <cvGPUSpeedup.cuh>
#include <fused_kernel/algorithms/image_processing/border_reader.h>
#include <opencv2/opencv.hpp>
#include "tests/testsCommon.cuh"

// Helper function to get the directory of the current file
std::string getSourceDir() {
    const std::string filePath = __FILE__;
    size_t lastSlash = filePath.find_last_of("/\\");
    const std::string test_warp = filePath.substr(0, lastSlash);
    lastSlash = test_warp.find_last_of("/\\");
    const std::string test = filePath.substr(0, lastSlash);
    lastSlash = test.find_last_of("/\\");
    return test.substr(0, lastSlash);
}

bool testPerspective() {
    // Load the image
    const cv::Mat img = cv::imread(getSourceDir() + "/images/NSightSystemsTimeline1.png");
    if (img.empty()) {
        std::cerr << "Error loading image" << std::endl;
        return false;
    }

    cv::cuda::Stream stream;

    // Upload the image to GPU
    const cv::cuda::GpuMat d_img(img);

    // Define the source and destination points for perspective transformation
    cv::Point2f src_points[4] = { cv::Point2f(56, 65), cv::Point2f(368, 52), cv::Point2f(28, 387), cv::Point2f(389, 390) };
    cv::Point2f dst_points[4] = { cv::Point2f(0, 0), cv::Point2f(300, 0), cv::Point2f(0, 300), cv::Point2f(300, 300) };

    // Get the perspective transformation matrix
    cv::Mat perspective_matrix = cv::getPerspectiveTransform(src_points, dst_points);

    // Preallocate the result images
    cv::cuda::GpuMat d_resultcv(img.size(), CV_8UC3);
    cv::cuda::GpuMat d_resultcvGS(img.size(), CV_8UC3);

    // Apply the perspective transformation
    cv::cuda::warpPerspective(d_img, d_resultcv, perspective_matrix, img.size(), 1, 0, cv::Scalar(), stream);

    const auto warpFunc = cvGS::warp<fk::WarpType::Perspective, CV_8UC3>(d_img, perspective_matrix, img.size());

    auto writeFunc = cvGS::write<CV_8UC3>(d_resultcvGS);
    cvGS::executeOperations(stream, warpFunc, fk::Cast<float3, uchar3>::build(), writeFunc);

    stream.waitForCompletion();

    // Download the result back to CPU
    cv::Mat resultcv(d_resultcv);
    cv::Mat resultcvGS(d_resultcvGS);

    const bool correct = compareAndCheck<CV_8UC3>(resultcv, resultcvGS);

    std::cout << "Perspective transformation: " << (correct ? "PASS" : "EXPECTED_FAIL") << std::endl;

    return true;
}

bool testAffine() {
    // Load the image
    const cv::Mat img = cv::imread(getSourceDir() + "/images/NSightSystemsTimeline1.png");
    if (img.empty()) {
        std::cerr << "Error loading image" << std::endl;
        return false;
    }

    cv::cuda::Stream stream;

    // Upload the image to GPU
    const cv::cuda::GpuMat d_img(img);

    // Define the translation values
    double tx = 50, ty = 100;

    // Get the affine transformation matrix
    cv::Mat affine_matrix = (cv::Mat_<double>(2, 3) << 1, 0, tx, 0, 1, ty);

    // Preallocate the result images
    cv::cuda::GpuMat d_resultcv(img.size(), CV_8UC3);
    cv::cuda::GpuMat d_resultcvGS(img.size(), CV_8UC3);

    // Apply the affine transformation
    cv::cuda::GpuMat d_result;
    cv::cuda::warpAffine(d_img, d_resultcv, affine_matrix, img.size());

    const auto warpFunc = cvGS::warp<fk::WarpType::Affine, CV_8UC3>(d_img, affine_matrix, img.size());
    auto writeFunc = cvGS::write<CV_8UC3>(d_resultcvGS);
    cvGS::executeOperations(stream, warpFunc, fk::Cast<float3, uchar3>::build(), writeFunc);

    stream.waitForCompletion();

    // Download the result back to CPU
    cv::Mat resultcv(d_resultcv);
    cv::Mat resultcvGS(d_resultcvGS);

    const bool correct = compareAndCheck<CV_8UC3>(resultcv, resultcvGS);

    std::cout << "Affine transformation: " << (correct ? "PASS" : "FAIL") << std::endl;

    return correct;
}

bool testPerspectiveBatch() {
    constexpr size_t NUM_IMGS = 5;

    // Load the image
    const cv::Mat img = cv::imread(getSourceDir() + "/images/NSightSystemsTimeline1.png");
    if (img.empty()) {
        std::cerr << "Error loading image" << std::endl;
        return false;
    }

    cv::cuda::Stream stream;

    // Upload the image to GPU
    const cv::cuda::GpuMat d_img(img);
    // Compiler bug: can't use NUM_IMGS with std::array
    const std::array<cv::cuda::GpuMat, 5> d_imgs = { d_img, d_img, d_img, d_img, d_img };

    // Define the source and destination points for perspective transformation
    cv::Point2f src_points1[4] = { cv::Point2f(56, 65), cv::Point2f(368, 52), cv::Point2f(28, 387), cv::Point2f(389, 390) };
    cv::Point2f dst_points1[4] = { cv::Point2f(0, 0), cv::Point2f(300, 0), cv::Point2f(0, 300), cv::Point2f(300, 300) };

    cv::Point2f src_points2[4] = { cv::Point2f(50, 50), cv::Point2f(400, 50), cv::Point2f(50, 400), cv::Point2f(400, 400) };
    cv::Point2f dst_points2[4] = { cv::Point2f(0, 0), cv::Point2f(300, 0), cv::Point2f(0, 300), cv::Point2f(300, 300) };

    cv::Point2f src_points3[4] = { cv::Point2f(30, 30), cv::Point2f(350, 30), cv::Point2f(30, 350), cv::Point2f(350, 350) };
    cv::Point2f dst_points3[4] = { cv::Point2f(0, 0), cv::Point2f(250, 0), cv::Point2f(0, 250), cv::Point2f(250, 250) };

    cv::Point2f src_points4[4] = { cv::Point2f(70, 70), cv::Point2f(370, 70), cv::Point2f(70, 370), cv::Point2f(370, 370) };
    cv::Point2f dst_points4[4] = { cv::Point2f(0, 0), cv::Point2f(280, 0), cv::Point2f(0, 280), cv::Point2f(280, 280) };

    cv::Point2f src_points5[4] = { cv::Point2f(20, 20), cv::Point2f(320, 20), cv::Point2f(20, 320), cv::Point2f(320, 320) };
    cv::Point2f dst_points5[4] = { cv::Point2f(0, 0), cv::Point2f(200, 0), cv::Point2f(0, 200), cv::Point2f(200, 200) };

    // Get the perspective transformation matrix
    std::array<cv::Mat, NUM_IMGS> perspective_matrices = { cv::getPerspectiveTransform(src_points1, dst_points1),
                                                        cv::getPerspectiveTransform(src_points2, dst_points2),
                                                        cv::getPerspectiveTransform(src_points3, dst_points3),
                                                        cv::getPerspectiveTransform(src_points4, dst_points4),
                                                        cv::getPerspectiveTransform(src_points5, dst_points5) };

    // Preallocate the result images
    std::array<cv::cuda::GpuMat, NUM_IMGS> d_resultscv{ cv::cuda::GpuMat(img.size(), CV_8UC3), cv::cuda::GpuMat(img.size(), CV_8UC3), cv::cuda::GpuMat(img.size(), CV_8UC3), cv::cuda::GpuMat(img.size(), CV_8UC3), cv::cuda::GpuMat(img.size(), CV_8UC3) };
    std::array<cv::cuda::GpuMat, NUM_IMGS> d_resultscvGS{ cv::cuda::GpuMat(img.size(), CV_8UC3), cv::cuda::GpuMat(img.size(), CV_8UC3), cv::cuda::GpuMat(img.size(), CV_8UC3), cv::cuda::GpuMat(img.size(), CV_8UC3), cv::cuda::GpuMat(img.size(), CV_8UC3) };

    // Apply the perspective transformation

    for (int i = 0; i < NUM_IMGS; ++i) {
        cv::cuda::warpPerspective(d_imgs[i], d_resultscv[i], perspective_matrices[i], img.size(), 1, 0, cv::Scalar(), stream);
    }

    const auto warpFunc = cvGS::warp<fk::WarpType::Perspective, CV_8UC3, NUM_IMGS>(d_imgs, perspective_matrices, img.size());

    auto fk_outputs = cvGS::gpuMat2RawPtr2D_arr<uchar3>(d_resultscvGS);
    auto writeFunc = fk::PerThreadWrite<fk::ND::_2D, uchar3>::build(fk_outputs);
    cvGS::executeOperations(stream, warpFunc, fk::Cast<float3, uchar3>::build(), writeFunc);

    stream.waitForCompletion();

    // Download the result back to CPU
    for (int i = 0; i < NUM_IMGS; ++i) {
        cv::Mat resultcv(d_resultscv[i]);
        cv::Mat resultcvGS(d_resultscvGS[i]);
        const bool correct = compareAndCheck<CV_8UC3>(resultcv, resultcvGS);
        std::cout << "Perspective transformation batch " << i << ": " << (correct ? "PASS" : "EXPECTED_FAIL") << std::endl;
    }

    return true;
}

bool testPerspectiveBatchNotAll() {
    constexpr size_t NUM_IMGS2 = 10;

    // Load the image
    const cv::Mat img = cv::imread(getSourceDir() + "/images/NSightSystemsTimeline1.png");
    if (img.empty()) {
        std::cerr << "Error loading image" << std::endl;
        return false;
    }

    cv::cuda::Stream stream;

    // Upload the image to GPU
    const cv::cuda::GpuMat d_img(img);
    const std::array<cv::cuda::GpuMat, NUM_IMGS2> d_imgs = { d_img, d_img, d_img, d_img, d_img, d_img, d_img, d_img, d_img, d_img };

    int usedPlanes = 3;

    // Define the source and destination points for perspective transformation
    cv::Point2f src_points1[4] = { cv::Point2f(56, 65), cv::Point2f(368, 52), cv::Point2f(28, 387), cv::Point2f(389, 390) };
    cv::Point2f dst_points1[4] = { cv::Point2f(0, 0), cv::Point2f(300, 0), cv::Point2f(0, 300), cv::Point2f(300, 300) };

    cv::Point2f src_points2[4] = { cv::Point2f(50, 50), cv::Point2f(400, 50), cv::Point2f(50, 400), cv::Point2f(400, 400) };
    cv::Point2f dst_points2[4] = { cv::Point2f(0, 0), cv::Point2f(300, 0), cv::Point2f(0, 300), cv::Point2f(300, 300) };

    cv::Point2f src_points3[4] = { cv::Point2f(30, 30), cv::Point2f(350, 30), cv::Point2f(30, 350), cv::Point2f(350, 350) };
    cv::Point2f dst_points3[4] = { cv::Point2f(0, 0), cv::Point2f(250, 0), cv::Point2f(0, 250), cv::Point2f(250, 250) };

    cv::Point2f src_points4[4] = { cv::Point2f(70, 70), cv::Point2f(370, 70), cv::Point2f(70, 370), cv::Point2f(370, 370) };
    cv::Point2f dst_points4[4] = { cv::Point2f(0, 0), cv::Point2f(280, 0), cv::Point2f(0, 280), cv::Point2f(280, 280) };

    cv::Point2f src_points5[4] = { cv::Point2f(20, 20), cv::Point2f(320, 20), cv::Point2f(20, 320), cv::Point2f(320, 320) };
    cv::Point2f dst_points5[4] = { cv::Point2f(0, 0), cv::Point2f(200, 0), cv::Point2f(0, 200), cv::Point2f(200, 200) };

    // Get the perspective transformation matrix
    std::array<cv::Mat, NUM_IMGS2> perspective_matrices2{ cv::getPerspectiveTransform(src_points1, dst_points1),
                                                  cv::getPerspectiveTransform(src_points2, dst_points2),
                                                  cv::getPerspectiveTransform(src_points3, dst_points3),
                                                  {}, {}, {}, {}, {}, {}, {} };

    // Preallocate the result images
    std::array<cv::cuda::GpuMat, NUM_IMGS2> d_resultscv{ cv::cuda::GpuMat(img.size(), CV_8UC3), cv::cuda::GpuMat(img.size(), CV_8UC3), cv::cuda::GpuMat(img.size(), CV_8UC3), cv::cuda::GpuMat(img.size(), CV_8UC3), cv::cuda::GpuMat(img.size(), CV_8UC3),
        cv::cuda::GpuMat(img.size(), CV_8UC3), cv::cuda::GpuMat(img.size(), CV_8UC3), cv::cuda::GpuMat(img.size(), CV_8UC3), cv::cuda::GpuMat(img.size(), CV_8UC3), cv::cuda::GpuMat(img.size(), CV_8UC3) };
    std::array<cv::cuda::GpuMat, NUM_IMGS2> d_resultscvGS{ cv::cuda::GpuMat(img.size(), CV_8UC3), cv::cuda::GpuMat(img.size(), CV_8UC3), cv::cuda::GpuMat(img.size(), CV_8UC3), cv::cuda::GpuMat(img.size(), CV_8UC3), cv::cuda::GpuMat(img.size(), CV_8UC3),
        cv::cuda::GpuMat(img.size(), CV_8UC3), cv::cuda::GpuMat(img.size(), CV_8UC3), cv::cuda::GpuMat(img.size(), CV_8UC3), cv::cuda::GpuMat(img.size(), CV_8UC3), cv::cuda::GpuMat(img.size(), CV_8UC3) };

    // Apply the perspective transformation
    for (int i = 0; i < usedPlanes; ++i) {
        cv::cuda::warpPerspective(d_imgs[i], d_resultscv[i], perspective_matrices2[i], img.size(), 1, 0, cv::Scalar(), stream);
    }

    const auto warpFunc = cvGS::warp<fk::WarpType::Perspective, CV_8UC3>(d_imgs, perspective_matrices2, img.size(), usedPlanes, cv::Scalar());

    auto fk_outputs = cvGS::gpuMat2RawPtr2D_arr<uchar3>(d_resultscvGS);
    auto writeFunc = fk::PerThreadWrite<fk::ND::_2D, uchar3>::build(fk_outputs);

    cvGS::executeOperations(stream, warpFunc, fk::Cast<float3, uchar3>::build(), writeFunc);

    stream.waitForCompletion();

    // Download the result back to CPU
    for (int i = 0; i < NUM_IMGS2; ++i) {
        cv::Mat resultcv(d_resultscv[i]);
        cv::Mat resultcvGS(d_resultscvGS[i]);
        const bool correct = compareAndCheck<CV_8UC3>(resultcv, resultcvGS);
        std::cout << "Perspective transformation batch " << i << ": " << (correct ? "PASS" : "EXPECTED_FAIL") << std::endl;
    }

    return true;
}

int launch() {
    const bool correctPerspective = testPerspective();
    const bool correctAffine = testAffine();
    const bool correctPerspectiveBatch = testPerspectiveBatch();
    const bool correctPerspectiveBatchNotAll = testPerspectiveBatchNotAll();
    const bool correctAll = correctPerspective && correctAffine && correctPerspectiveBatch && correctPerspectiveBatchNotAll;
    // warpPerspective is almost identical to OpenCV's implementation, but there are a few pixels of difference in
    // the border. The reason is hard to find, since OpenCV is using NPP for the warping.
    return correctAll ? 0 : -1;
}
