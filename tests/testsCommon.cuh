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

#include "tests/testUtils.cuh"
#include <cv2cuda_types.cuh>
#include <fused_kernel/core/utils/vlimits.h>

#include <sstream>
#include <fstream>
#include <unordered_map>
#include <array>
#include <chrono>

#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>

template <size_t START_VALUE, size_t INCREMENT, std::size_t... Is>
constexpr std::array<size_t, sizeof...(Is)> generate_sequence(std::index_sequence<Is...>) {
    return std::array<size_t, sizeof...(Is)>{(START_VALUE + (INCREMENT * Is))...};
}

template <size_t START_VALUE, size_t INCREMENT, size_t NUM_ELEMS>
constexpr std::array<size_t, NUM_ELEMS> arrayIndexSecuence = generate_sequence<START_VALUE, INCREMENT>(std::make_index_sequence<NUM_ELEMS>{});

template <int T>
bool checkResults(int NUM_ELEMS_X, int NUM_ELEMS_Y, cv::Mat& h_comparison1C) {
    if constexpr (std::is_floating_point_v<CUDA_T(T)>) {
        cv::Mat h_comparison(NUM_ELEMS_Y, NUM_ELEMS_X, T);
        cv::Mat maxError(NUM_ELEMS_Y, NUM_ELEMS_X, T, 0.0001);
        cv::compare(h_comparison1C, maxError, h_comparison, cv::CMP_GT);

#ifdef CVGS_DEBUG
        for (int y = 0; y < h_comparison1C.rows; y++) {
            for (int x = 0; x < h_comparison1C.cols; x++) {
                CUDA_T(T) value = h_comparison1C.at<CUDA_T(T)>(y, x);
                if (value > 0.00001) {
                    std::cout << "(" << x << "," << y << ")= " << value << ";" << std::endl;
                }
            }
            std::cout << std::endl;
        }
#endif

        int errors = cv::countNonZero(h_comparison);
        return errors == 0;
    } else {
        return cv::sum(h_comparison1C)[0] == 0;
    }
    
}

template <int T>
bool compareAndCheck(int NUM_ELEMS_X, int NUM_ELEMS_Y, cv::Mat& cvVersion, cv::Mat& cvGSVersion) {
    bool passed = true;
    cv::Mat diff = cv::abs(cvVersion - cvGSVersion);
    std::vector<cv::Mat> h_comparison1C(CV_MAT_CN(T));
    cv::split(diff, h_comparison1C);
    #ifdef CVGS_DEBUG
    for (int y = 0; y < cvVersion.rows; y++) {
        for (int x = 0; x < cvVersion.cols; x++) {
            std::cout << "OpenCV (" << x << "," << y << ")= " << cvVersion.at<CUDA_T(T)>(y, x) << ";" << std::endl;
            std::cout << "cvGPUSpeedup (" << x << "," << y << ")= " << cvGSVersion.at<CUDA_T(T)>(y, x) << ";" << std::endl;
        }
    }
    #endif
    for (int i = 0; i < CV_MAT_CN(T); i++) {
        passed &= checkResults<CV_MAT_DEPTH(T)>(NUM_ELEMS_X, NUM_ELEMS_Y, h_comparison1C.at(i));
    }
    return passed;
}

template <int T>
bool compareAndCheck(cv::Mat& cvVersion, cv::Mat& cvGSVersion) {
    if (cvVersion.size != cvGSVersion.size) {
        throw std::runtime_error("The size of the images is different");
    }
    const int NUM_ELEMS_X = cvVersion.cols;
    const int NUM_ELEMS_Y = cvVersion.rows;
    bool passed = true;
    cv::Mat diff = cv::abs(cvVersion - cvGSVersion);
    if constexpr (CV_MAT_CN(T) == 1) {
        passed &= checkResults<CV_MAT_DEPTH(T)>(NUM_ELEMS_X, NUM_ELEMS_Y, diff);
    } else {
        std::vector<cv::Mat> h_comparison1C(CV_MAT_CN(T));
        cv::split(diff, h_comparison1C);

        for (int i = 0; i < CV_MAT_CN(T); i++) {
            passed &= checkResults<CV_MAT_DEPTH(T)>(NUM_ELEMS_X, NUM_ELEMS_Y, h_comparison1C.at(i));
        }
    }
    return passed;
}

struct BenchmarkResultsNumbers {
    float OCVelapsedTimeMax;
    float OCVelapsedTimeMin;
    float OCVelapsedTimeAcum;
    float cvGSelapsedTimeMax;
    float cvGSelapsedTimeMin;
    float cvGSelapsedTimeAcum;
};

template <int ITERS>
struct BenchmarkTemp {
    BenchmarkResultsNumbers resF;
    std::chrono::steady_clock::time_point startTime;
    std::array<float, ITERS> OCVelapsedTime;
    std::array<float, ITERS> cvGSelapsedTime;
};

#ifdef ENABLE_BENCHMARK
std::unordered_map<std::string, std::stringstream> benchmarkResultsText;
std::unordered_map<std::string, std::ofstream> currentFile;
// Select the path where to write the benchmark files
const std::string path{ "" };

constexpr int ITERS = 100;
constexpr int ITERS_W = 1;
bool warmup = false;

template <size_t ITERATIONS>
float computeVariance(const float& mean, const std::array<float, ITERATIONS>& times) {
    float sumOfDiff = 0.f;
    for (int i = 0; i < ITERATIONS; i++) {
        const float diff = times[i] - mean;
        sumOfDiff += (diff * diff);
    }
    return sumOfDiff / (ITERATIONS - 1);
}

template <int CV_INPUT_TYPE, int CV_OUTPUT_TYPE, int BATCH, int ITERATIONS, int NUM_BATCH_VALUES, const std::array<size_t, NUM_BATCH_VALUES>& batchValues>
inline void processExecution(const BenchmarkResultsNumbers& resF,
                             const std::string& functionName,
                             const std::array<float, ITERS>& OCVelapsedTime,
                             const std::array<float, ITERS>& cvGSelapsedTime,
                             const std::string& variableDimension) {

    // Create 2D Table for changing types and changing batch
    const std::string fileName = functionName + std::string(".csv");
    if constexpr (BATCH == batchValues[0]) {
        if (currentFile.find(fileName) == currentFile.end()) {
            currentFile[fileName].open(path + fileName);
        }
        currentFile[fileName]
            << variableDimension << " ("
            << cvTypeToString<CV_INPUT_TYPE>() << "X"
            << cvTypeToString<CV_OUTPUT_TYPE>() << ")";
        currentFile[fileName] << ", OpenCV MeanTime";
        currentFile[fileName] << ", OpenCV TimeVariance";
        currentFile[fileName] << ", OpenCV MaxTime";
        currentFile[fileName] << ", OpenCV MinTime";
        currentFile[fileName] << ", cvGS MeanTime";
        currentFile[fileName] << ", cvGS TimeVariance";
        currentFile[fileName] << ", cvGS MaxTime";
        currentFile[fileName] << ", cvGS MinTime";
        currentFile[fileName] << ", Mean Speedup";
        currentFile[fileName] << std::endl;
    }

    const bool mustStore = currentFile.find(fileName) != currentFile.end();
    if (mustStore) {
        const float ocvMean = resF.OCVelapsedTimeAcum / ITERATIONS;
        const float cvgsMean = resF.cvGSelapsedTimeAcum / ITERATIONS;
        const float ocvVariance = computeVariance(ocvMean, OCVelapsedTime);
        const float cvgsVariance = computeVariance(cvgsMean, cvGSelapsedTime);
        float meanSpeedup{ 0.f };
        for (int i = 0; i < ITERS; i++) {
            meanSpeedup += OCVelapsedTime[i] / cvGSelapsedTime[i];
        }
        meanSpeedup /= ITERS;

        currentFile[fileName] << BATCH;
        currentFile[fileName] << ", " << ocvMean;
        currentFile[fileName] << ", " << computeVariance(ocvMean, OCVelapsedTime);
        currentFile[fileName] << ", " << resF.OCVelapsedTimeMax;
        currentFile[fileName] << ", " << resF.OCVelapsedTimeMin;
        currentFile[fileName] << ", " << cvgsMean;
        currentFile[fileName] << ", " << computeVariance(cvgsMean, cvGSelapsedTime);
        currentFile[fileName] << ", " << resF.cvGSelapsedTimeMax;
        currentFile[fileName] << ", " << resF.cvGSelapsedTimeMin;
        currentFile[fileName] << ", " << meanSpeedup;
        currentFile[fileName] << std::endl;
    }
}

template <int CV_INPUT_TYPE, int CV_OUTPUT_TYPE, int ITERS, int BATCH, int FIRST_VALUE, int INCREMENT, int NUM_EXPERIMENTS>
BenchmarkTemp<ITERS> initCPUBenchmark(const std::string& functionName, const std::string& variableDimension) {
    std::cout << "Executing " << functionName << " fusing " << BATCH << " operations. " << ((BATCH - FIRST_VALUE) / INCREMENT) + 1 << "/" << NUM_EXPERIMENTS << std::endl;

    BenchmarkTemp<ITERS> benchmarkTemp;
    benchmarkTemp.resF.OCVelapsedTimeMax = fk::minValue<float>;
    benchmarkTemp.resF.OCVelapsedTimeMin = fk::maxValue<float>;
    benchmarkTemp.resF.OCVelapsedTimeAcum = 0.f;
    benchmarkTemp.resF.cvGSelapsedTimeMax = fk::minValue<float>;
    benchmarkTemp.resF.cvGSelapsedTimeMin = fk::maxValue<float>;
    benchmarkTemp.resF.cvGSelapsedTimeAcum = 0.f;

    return benchmarkTemp;
}

template <int ITERS>
void startOCV_CPU(BenchmarkTemp<ITERS>& benchmarkTemp) {
    benchmarkTemp.startTime = std::chrono::steady_clock::now();
}

template <int ITERS>
void stopOCV_startcvGS_CPU(BenchmarkTemp<ITERS>& benchmarkTemp, const int& i) {
    const auto endTime = std::chrono::steady_clock::now();
    const std::chrono::duration<float, std::milli> elapsedTime = endTime - benchmarkTemp.startTime;
    benchmarkTemp.OCVelapsedTime[i] = elapsedTime.count();
    benchmarkTemp.resF.OCVelapsedTimeMax = benchmarkTemp.resF.OCVelapsedTimeMax < benchmarkTemp.OCVelapsedTime[i] ? benchmarkTemp.OCVelapsedTime[i] : benchmarkTemp.resF.OCVelapsedTimeMax;
    benchmarkTemp.resF.OCVelapsedTimeMin = benchmarkTemp.resF.OCVelapsedTimeMin > benchmarkTemp.OCVelapsedTime[i] ? benchmarkTemp.OCVelapsedTime[i] : benchmarkTemp.resF.OCVelapsedTimeMin;
    benchmarkTemp.resF.OCVelapsedTimeAcum += benchmarkTemp.OCVelapsedTime[i];

    benchmarkTemp.startTime = std::chrono::steady_clock::now();
}

template <int ITERS>
void stopcvGS_CPU(BenchmarkTemp<ITERS>& benchmarkTemp, const int& i) {
    const auto endTime = std::chrono::steady_clock::now();
    const std::chrono::duration<float, std::milli> elapsedTime = endTime - benchmarkTemp.startTime;
    benchmarkTemp.cvGSelapsedTime[i] = elapsedTime.count();
    benchmarkTemp.resF.cvGSelapsedTimeMax = benchmarkTemp.resF.cvGSelapsedTimeMax < benchmarkTemp.cvGSelapsedTime[i] ? benchmarkTemp.cvGSelapsedTime[i] : benchmarkTemp.resF.cvGSelapsedTimeMax;
    benchmarkTemp.resF.cvGSelapsedTimeMin = benchmarkTemp.resF.cvGSelapsedTimeMin > benchmarkTemp.cvGSelapsedTime[i] ? benchmarkTemp.cvGSelapsedTime[i] : benchmarkTemp.resF.cvGSelapsedTimeMin;
    benchmarkTemp.resF.cvGSelapsedTimeAcum += benchmarkTemp.cvGSelapsedTime[i];
}

void stopCPUBenchmark() {
    for (auto&& [_, file] : currentFile) {
        file.close();
    }
}

#else

template <int ITERS>
void startOCV_CPU(BenchmarkTemp<ITERS>& benchmarkTemp) {}

template <int ITERS>
void stopOCV_startcvGS_CPU(BenchmarkTemp<ITERS>& benchmarkTemp, const int& i) {}

template <int ITERS>
void stopcvGS_CPU(BenchmarkTemp<ITERS>& benchmarkTemp, const int& i) {}

void stopCPUBenchmark() {}

#endif

#ifdef ENABLE_BENCHMARK
#define START_OCV_BENCHMARK \
std::cout << "Executing " << __func__ << " fusing " << BATCH << " operations. " << (BATCH - FIRST_VALUE)/INCREMENT << "/" << NUM_EXPERIMENTS << std::endl; \
cudaEvent_t start, stop; \
BenchmarkResultsNumbers resF; \
resF.OCVelapsedTimeMax = fk::minValue<float>; \
resF.OCVelapsedTimeMin = fk::maxValue<float>; \
resF.OCVelapsedTimeAcum = 0.f; \
resF.cvGSelapsedTimeMax = fk::minValue<float>; \
resF.cvGSelapsedTimeMin = fk::maxValue<float>; \
resF.cvGSelapsedTimeAcum = 0.f; \
cudaStream_t stream = cv::cuda::StreamAccessor::getStream(cv_stream); \
gpuErrchk(cudaEventCreate(&start)); \
gpuErrchk(cudaEventCreate(&stop)); \
std::array<float, ITERS> OCVelapsedTime; \
std::array<float, ITERS> cvGSelapsedTime; \
for (int i = 0; i < ITERS; i++) { \
    gpuErrchk(cudaEventRecord(start, stream));
#else
#define START_OCV_BENCHMARK
#endif

#ifdef ENABLE_BENCHMARK
#define STOP_OCV_START_CVGS_BENCHMARK \
gpuErrchk(cudaEventRecord(stop, stream)); \
gpuErrchk(cudaEventSynchronize(stop)); \
gpuErrchk(cudaEventElapsedTime(&OCVelapsedTime[i], start, stop)); \
resF.OCVelapsedTimeMax = resF.OCVelapsedTimeMax < OCVelapsedTime[i] ? OCVelapsedTime[i] : resF.OCVelapsedTimeMax; \
resF.OCVelapsedTimeMin = resF.OCVelapsedTimeMin > OCVelapsedTime[i] ? OCVelapsedTime[i] : resF.OCVelapsedTimeMin; \
resF.OCVelapsedTimeAcum += OCVelapsedTime[i]; \
gpuErrchk(cudaEventRecord(start, stream));
#else
#define STOP_OCV_START_CVGS_BENCHMARK
#endif

#ifdef ENABLE_BENCHMARK
#define STOP_CVGS_BENCHMARK \
gpuErrchk(cudaEventRecord(stop, stream)); \
gpuErrchk(cudaEventSynchronize(stop)); \
gpuErrchk(cudaEventElapsedTime(&cvGSelapsedTime[i], start, stop)); \
resF.cvGSelapsedTimeMax = resF.cvGSelapsedTimeMax < cvGSelapsedTime[i] ? cvGSelapsedTime[i] : resF.cvGSelapsedTimeMax; \
resF.cvGSelapsedTimeMin = resF.cvGSelapsedTimeMin > cvGSelapsedTime[i] ? cvGSelapsedTime[i] : resF.cvGSelapsedTimeMin; \
resF.cvGSelapsedTimeAcum += cvGSelapsedTime[i]; \
if (warmup) break; \
} \
processExecution<CV_TYPE_I, CV_TYPE_O, BATCH, ITERS, batchValues.size(), batchValues>(resF, __func__, OCVelapsedTime, cvGSelapsedTime, VARIABLE_DIMENSION);
#else
#define STOP_CVGS_BENCHMARK
#endif

#ifdef ENABLE_BENCHMARK
#define CLOSE_BENCHMARK \
for (auto&& [_, file] : currentFile) { \
    file.close(); \
}
#else
#define CLOSE_BENCHMARK
#endif