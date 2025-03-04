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
#include <opencv2/opencv.hpp>

bool testPerspective() {
    // Load the image
    const cv::Mat img = cv::imread("E:/GitHub/cvGPUSpeedup/images/NSightSystemsTimeline1.png");
    if (img.empty()) {
        std::cerr << "Error loading image" << std::endl;
        return -1;
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

    cv::Mat inverted_perspective_matrix;
    invert(perspective_matrix, inverted_perspective_matrix);

    const auto warpFunc = cvGS::warp<fk::WarpType::Perspective, CV_8UC3>(d_img, inverted_perspective_matrix, img.size());

    bool correct{ true };
    /*const double* const rawMat = perspective_matrix.ptr<double>();
    correct &= std::abs(static_cast<float>(rawMat[0]) - warpFunc.params.transformMatrix.data[0][0]) < 0.001;
    correct &= std::abs(static_cast<float>(rawMat[1]) - warpFunc.params.transformMatrix.data[0][1]) < 0.001;
    correct &= std::abs(static_cast<float>(rawMat[2]) - warpFunc.params.transformMatrix.data[0][2]) < 0.001;
    correct &= std::abs(static_cast<float>(rawMat[3]) - warpFunc.params.transformMatrix.data[1][0]) < 0.001;
    correct &= std::abs(static_cast<float>(rawMat[4]) - warpFunc.params.transformMatrix.data[1][1]) < 0.001;
    correct &= std::abs(static_cast<float>(rawMat[5]) - warpFunc.params.transformMatrix.data[1][2]) < 0.001;
    correct &= std::abs(static_cast<float>(rawMat[6]) - warpFunc.params.transformMatrix.data[2][0]) < 0.001;
    correct &= std::abs(static_cast<float>(rawMat[7]) - warpFunc.params.transformMatrix.data[2][1]) < 0.001;
    correct &= std::abs(static_cast<float>(rawMat[8]) - warpFunc.params.transformMatrix.data[2][2]) < 0.001;*/

    auto writeFunc = cvGS::write<CV_8UC3>(d_resultcvGS);
    cvGS::executeOperations(stream, warpFunc, fk::Cast<float3, uchar3>::build(), writeFunc);

    stream.waitForCompletion();

    // Download the result back to CPU
    cv::Mat resultcv(d_resultcv);
    cv::Mat resultcvGS(d_resultcvGS);

    return correct;
}

bool testAffine() {
    // Load the image
    const cv::Mat img = cv::imread("E:/GitHub/cvGPUSpeedup/images/NSightSystemsTimeline1.png");
    if (img.empty()) {
        std::cerr << "Error loading image" << std::endl;
        return -1;
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

    cv::Mat inverted_affine_matrix;
    cv::invertAffineTransform(affine_matrix, inverted_affine_matrix);

    const auto warpFunc = cvGS::warp<fk::WarpType::Affine, CV_8UC3>(d_img, inverted_affine_matrix, img.size());
    auto writeFunc = cvGS::write<CV_8UC3>(d_resultcvGS);
    cvGS::executeOperations(stream, warpFunc, fk::Cast<float3, uchar3>::build(), writeFunc);

    stream.waitForCompletion();

    // Download the result back to CPU
    cv::Mat resultcv(d_resultcv);
    cv::Mat resultcvGS(d_resultcvGS);

    return true;
}

int launch() {
    return testPerspective() && testAffine() ? 0 : -1;
}
