/* Copyright 2025 Grup Mediapro, S.L.U. (Oscar Amoros Huguet)
   Copyright 2023-2025 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#pragma once

#include <cvGPUSpeedupHelpers.cuh>
#include <fused_kernel/fused_kernel.cuh>
#include <fused_kernel/core/data/circular_tensor.cuh>
#include <fused_kernel/algorithms/image_processing/resize.cuh>
#include <fused_kernel/algorithms/image_processing/color_conversion.cuh>
#include <fused_kernel/algorithms/image_processing/crop.cuh>
#include <fused_kernel/algorithms/image_processing/border_reader.cuh>
#include <fused_kernel/algorithms/image_processing/warping.cuh>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

namespace cvGS {

enum AspectRatio { PRESERVE_AR = 0, IGNORE_AR = 1, PRESERVE_AR_RN_EVEN = 2, PRESERVE_AR_LEFT = 3 };

template <typename T>
inline constexpr fk::Ptr2D<T> gpuMat2Ptr2D(const cv::cuda::GpuMat& source) {
    const fk::Ptr2D<T> temp(reinterpret_cast<T*>(source.data), source.cols, source.rows, (uint)source.step);
    return temp;
}

template <typename T>
inline constexpr fk::RawPtr<fk::_2D, T> gpuMat2RawPtr2D(const cv::cuda::GpuMat& source) {
    const fk::RawPtr<fk::_2D, T> temp{ (T*)source.data, {static_cast<uint>(source.cols), static_cast<uint>(source.rows), static_cast<uint>(source.step)} };
    return temp;
}

template <typename T, int Batch>
inline constexpr std::array<fk::Ptr2D<T>, Batch> gpuMat2Ptr2D_arr(const std::array<cv::cuda::GpuMat, Batch>& source) {
    std::array<fk::Ptr2D<T>, Batch> temp;
    std::transform(source.begin(), source.end(), temp.begin(),
                        [](const cv::cuda::GpuMat& i) { return gpuMat2Ptr2D<T>(i); });
    return temp;
}

template <typename T, size_t Batch>
inline constexpr std::array<fk::RawPtr<fk::_2D, T>, Batch> gpuMat2RawPtr2D_arr(const std::array<cv::cuda::GpuMat, Batch>& source, const int& usedElems = Batch) {
    std::array<fk::RawPtr<fk::_2D, T>, Batch> temp;
    if (usedElems == Batch) {
        std::transform(source.begin(), source.end(), temp.begin(),
            [](const cv::cuda::GpuMat& i) { return gpuMat2RawPtr2D<T>(i); });
    } else {
        std::transform(source.begin(), source.begin() + usedElems, temp.begin(),
            [](const cv::cuda::GpuMat& i) { return gpuMat2RawPtr2D<T>(i); });
    }
    return temp;
}

template <typename T>
inline constexpr fk::Tensor<T> gpuMat2Tensor(const cv::cuda::GpuMat& source, const cv::Size& planeDims, const int& colorPlanes) {
    const fk::Tensor<T> t_output((T*)source.data, planeDims.width, planeDims.height, source.rows, colorPlanes);
    return t_output;
}


template <int I, int O>
inline constexpr auto convertTo() {
    static_assert(fk::cn<CUDA_T(I)> == fk::cn<CUDA_T(O)>, "convertTo does not support changing the number of channels, neither in cvGS nor in OpenCV. Please, use cvGS::cvtColor instead.");
    return fk::Unary<fk::SaturateCast<CUDA_T(I), CUDA_T(O)>>{};
}

template <int I, int O>
inline constexpr auto convertTo(float alpha) {
    static_assert(fk::cn<CUDA_T(I)> == fk::cn<CUDA_T(O)>, "convertTo does not support changing the number of channels, neither in cvGS nor in OpenCV. Please, use cvGS::cvtColor instead.");

    using InputBase = typename fk::VectorTraits<CUDA_T(I)>::base;
    using OutputBase = typename fk::VectorTraits<CUDA_T(O)>::base;
    constexpr bool outputIsIntegral = std::is_integral_v<fk::VBase<CUDA_T(O)>>;
    using FloatType = std::conditional_t<outputIsIntegral, fk::VectorType_t<float, fk::cn<CUDA_T(O)>>, CUDA_T(O)>;
    const auto alphaVec = fk::make_set<FloatType>(alpha);

    if constexpr (outputIsIntegral) {
        using FirstOp = fk::SaturateCast<CUDA_T(I), FloatType>;
        using SecondOp = fk::Mul<FloatType>;
        using ThirdOp = fk::SaturateCast<FloatType, CUDA_T(O)>;

        return FirstOp::build().then(SecondOp::build(alphaVec)).then(ThirdOp::build());
    } else {
        using FirstOp = fk::SaturateCast<CUDA_T(I), CUDA_T(O)>;
        using SecondOp = fk::Mul<CUDA_T(O)>;

        return FirstOp::build().then(SecondOp::build(alphaVec));
    }
}

template <int I, int O>
inline constexpr auto convertTo(float alpha, float beta) {
    static_assert(fk::cn<CUDA_T(I)> == fk::cn<CUDA_T(O)>, "convertTo does not support changing the number of channels, neither in cvGS nor in OpenCV. Please, use cvGS::cvtColor instead.");

    using InputBase = typename fk::VectorTraits<CUDA_T(I)>::base;
    using OutputBase = typename fk::VectorTraits<CUDA_T(O)>::base;
    constexpr bool outputIsIntegral = std::is_integral_v<fk::VBase<CUDA_T(O)>>;
    using FloatType = std::conditional_t<outputIsIntegral, fk::VectorType_t<float, fk::cn<CUDA_T(O)>>, CUDA_T(O)>;
    const auto alphaVec = fk::make_set<FloatType>(alpha);
    const auto betaVec = fk::make_set<FloatType>(beta);

    if constexpr (outputIsIntegral) {
        using FirstOp = fk::SaturateCast<CUDA_T(I), FloatType>;
        using SecondOp = fk::Mul<FloatType>;
        using ThirdOp = fk::Add<FloatType>;
        using FourthOp = fk::SaturateCast<FloatType, CUDA_T(O)>;
        
        return FirstOp::build().then(SecondOp::build(alphaVec)).then(ThirdOp::build(betaVec)).then(FourthOp::build());
    } else {
        using FirstOp = fk::SaturateCast<CUDA_T(I), CUDA_T(O)>;
        using SecondOp = fk::Mul<CUDA_T(O)>;
        using ThirdOp = fk::Add<CUDA_T(O)>;

        return fk::FusedOperation<FirstOp, SecondOp, ThirdOp>::build({ { { alphaVec, { betaVec } } } });
    }
}

template <int I>
inline constexpr auto multiply(const cv::Scalar& src2) {
    return fk::Binary<fk::Mul<CUDA_T(I)>> { cvScalar2CUDAV<I>::get(src2) };
}

template <int I>
inline constexpr auto subtract(const cv::Scalar& src2) {
    return fk::Binary<fk::Sub<CUDA_T(I)>> { cvScalar2CUDAV<I>::get(src2) };
}

template <int I>
inline constexpr auto divide(const cv::Scalar& src2) {
    return fk::Binary<fk::Div<CUDA_T(I)>> { cvScalar2CUDAV<I>::get(src2) };
}

template <int I>
inline constexpr auto add(const cv::Scalar& src2) {
    return fk::Binary<fk::Add<CUDA_T(I)>> { cvScalar2CUDAV<I>::get(src2) };
}

template <cv::ColorConversionCodes CODE, int I, int O = I>
inline constexpr auto cvtColor() {
    static_assert((CV_MAT_DEPTH(I) == CV_8U || CV_MAT_DEPTH(I) == CV_16U || CV_MAT_DEPTH(I) == CV_32F) &&
                  (CV_MAT_DEPTH(O) == CV_8U || CV_MAT_DEPTH(O) == CV_16U || CV_MAT_DEPTH(O) == CV_32F),
                  "Wrong CV_TYPE_DEPTH, it has to be CV_8U, or CV_16U or CV_32F");
    static_assert(isSupportedColorConversion<CODE>, "Color conversion type not supported yet.");
    using InputType = CUDA_T(I);
    using OutputType = CUDA_T(O);

    return fk::Unary<fk::ColorConversion<(fk::ColorConversionCodes)CODE, InputType, OutputType>>{};
}

template <int O>
inline constexpr auto split(const std::vector<cv::cuda::GpuMat>& output) {
    std::vector<fk::Ptr2D<BASE_CUDA_T(O)>> fk_output;
    for (auto& mat : output) {
        fk_output.push_back(gpuMat2Ptr2D<BASE_CUDA_T(O)>(mat));
    }
    return fk::SplitWrite<fk::_2D, CUDA_T(O)>::build(fk_output);
}

template <int O, size_t N>
inline constexpr auto split(const std::array<std::vector<cv::cuda::GpuMat>, N>& output) {
    std::array<std::vector<fk::Ptr2D<BASE_CUDA_T(O)>>, N> fkOutput{};
    for (int i = 0; i < N; ++i) {
        std::vector<fk::Ptr2D<BASE_CUDA_T(O)>> fk_output;
        for (auto& mat : output[i]) {
            fk_output.push_back(gpuMat2Ptr2D<BASE_CUDA_T(O)>(mat));
        }
        fkOutput[i] = fk_output;
    }
    return fk::SplitWrite<fk::_2D, CUDA_T(O)>::build(fkOutput);
}

template <int O>
inline constexpr auto split(const cv::cuda::GpuMat& output, const cv::Size& planeDims) {
    assert(output.cols % (planeDims.width * planeDims.height) == 0 && output.cols / (planeDims.width * planeDims.height) == CV_MAT_CN(O) &&
    "Each row of the GpuMap should contain as many planes as width / (planeDims.width * planeDims.height)");

    return fk::Write<fk::TensorSplit<CUDA_T(O)>> {
        gpuMat2Tensor<BASE_CUDA_T(O)>(output, planeDims, CV_MAT_CN(O)).ptr()};
}

template <int O>
inline constexpr auto split(const fk::RawPtr<fk::_3D, typename fk::VectorTraits<CUDA_T(O)>::base>& output) {
    return fk::Write<fk::TensorSplit<CUDA_T(O)>> {output};
}

template <int O>
inline constexpr auto splitT(const fk::RawPtr<fk::T3D, typename fk::VectorTraits<CUDA_T(O)>::base>& output) {
    return fk::Write<fk::TensorTSplit<CUDA_T(O)>> {output};
}

template <int INTER_F>
inline const auto resize(const cv::Size& dsize) {
    return fk::Resize<static_cast<fk::InterpolationType>(INTER_F)>::build(fk::Size(dsize.width, dsize.height));
}

template <int T, int INTER_F>
inline const auto resize(const cv::cuda::GpuMat& input, const cv::Size& dsize, double fx, double fy) {
    static_assert(isSupportedInterpolation<INTER_F>, "Interpolation type not supported yet.");

    const fk::RawPtr<fk::_2D, CUDA_T(T)> fk_input = gpuMat2Ptr2D<CUDA_T(T)>(input);
    const fk::Size dSize{ dsize.width, dsize.height };
    return fk::Resize<(fk::InterpolationType)INTER_F>::build(fk_input, dSize, fx, fy);
}

template <int T, int INTER_F, int NPtr, AspectRatio AR_ = IGNORE_AR>
inline const auto resize(const std::array<cv::cuda::GpuMat, NPtr>& input,
                         const cv::Size& dsize, const int& usedPlanes,
                         const cv::Scalar& backgroundValue_ = cvScalar_set<CV_MAKETYPE(CV_32F, CV_MAT_CN(T))>(0)) {
    static_assert(isSupportedInterpolation<INTER_F>, "Interpolation type not supported yet.");

    constexpr fk::AspectRatio AR = static_cast<fk::AspectRatio>(AR_);
    constexpr fk::InterpolationType IType = static_cast<fk::InterpolationType>(INTER_F);

    constexpr int defaultType = CV_MAKETYPE(CV_32F, CV_MAT_CN(T));
    const std::array<fk::RawPtr<fk::_2D, CUDA_T(T)>, NPtr> fk_input{ gpuMat2RawPtr2D_arr<CUDA_T(T), NPtr>(input) };

    using PixelReadOp = fk::PerThreadRead<fk::_2D, CUDA_T(T)>;
    using O = CUDA_T(defaultType);
    const O backgroundValue = cvScalar2CUDAV<defaultType>::get(backgroundValue_);
    
    const auto readOP = PixelReadOp::build_batch(fk_input);
    const auto sizeArr = fk::make_set_std_array<NPtr>(fk::Size(dsize.width, dsize.height));
    const auto backgroundArr = fk::make_set_std_array<NPtr>(backgroundValue);
    using Resize = fk::Resize<IType, AR, fk::Read<PixelReadOp>>;
    if constexpr (AR != fk::IGNORE_AR) {
        const auto resizeDFs = Resize::build_batch(readOP, sizeArr, backgroundArr);
        return fk::BatchRead<NPtr, fk::CONDITIONAL_WITH_DEFAULT>::build(resizeDFs, usedPlanes, backgroundValue);
    } else {
        const auto resizeDFs = Resize::build_batch(readOP, sizeArr);
        return fk::BatchRead<NPtr, fk::CONDITIONAL_WITH_DEFAULT>::build(resizeDFs, usedPlanes, backgroundValue);
    }
}

inline constexpr auto crop(const cv::Rect2d& rect) {
    return fk::Crop<>::build(fk::Rect(static_cast<uint>(rect.x), static_cast<uint>(rect.y), static_cast<int>(rect.width), static_cast<int>(rect.height)));
}

template <int BATCH>
inline constexpr auto crop(const std::array<cv::Rect2d, BATCH>& rects) {
    std::array<fk::Rect, BATCH> fk_rects{};

    for (int i = 0; i < BATCH; ++i) {
        const auto tmp = rects[i];
        fk_rects[i] = fk::Rect(static_cast<uint>(tmp.x), static_cast<uint>(tmp.y), static_cast<int>(tmp.width), static_cast<int>(tmp.height));
    }
    return fk::Crop<>::build(fk_rects);
}

template <typename BackIOp>
inline constexpr auto crop(const BackIOp& backIOp, const cv::Rect2d& rect) {
    return fk::Crop<BackIOp>::build(backIOp, fk::Rect(static_cast<uint>(rect.x), static_cast<uint>(rect.y), static_cast<int>(rect.width), static_cast<int>(rect.height)));
}

namespace internal {

inline constexpr auto warp_getWarpingAffineParameters(const double* const tm_raw, const cv::Size& dstSize){
    return fk::WarpingParameters<fk::WarpType::Affine>{{{
        { static_cast<float>(tm_raw[0]), static_cast<float>(tm_raw[1]), static_cast<float>(tm_raw[2]) },
        { static_cast<float>(tm_raw[3]), static_cast<float>(tm_raw[4]), static_cast<float>(tm_raw[5]) }}},
        fk::Size(dstSize.width, dstSize.height) };
}

inline constexpr auto warp_getWarpingPerspectiveParameters(const double* const tm_raw, const cv::Size& dstSize) {
    return fk::WarpingParameters<fk::WarpType::Perspective>{{{
        { static_cast<float>(tm_raw[0]), static_cast<float>(tm_raw[1]), static_cast<float>(tm_raw[2]) },
        { static_cast<float>(tm_raw[3]), static_cast<float>(tm_raw[4]), static_cast<float>(tm_raw[5]) },
        { static_cast<float>(tm_raw[6]), static_cast<float>(tm_raw[7]), static_cast<float>(tm_raw[8]) }}},
        fk::Size(dstSize.width, dstSize.height) };
}
} // namespace internal

template <enum fk::WarpType WT, int InputType = CV_8UC3>
inline constexpr auto warp(const cv::cuda::GpuMat& input, const cv::Mat& transform_matrix, const cv::Size& dstSize) {
    if (InputType != input.type()) {
        throw std::runtime_error("Input type does not match the input type of the operation.");
    }
    if (transform_matrix.type() != CV_64FC1) {
        throw std::runtime_error("Transform matrix type should be CV_64FC1.");
    }
    const auto read = fk::PerThreadRead<fk::_2D, CUDA_T(InputType)>::build(fk::RawPtr<fk::_2D, CUDA_T(InputType)>{ (CUDA_T(InputType)*)input.data, { static_cast<uint>(input.cols), static_cast<uint>(input.rows), static_cast<uint>(input.step) } });
    if constexpr (WT == fk::WarpType::Affine) {
        cv::Mat inverse_transform_matrix;
        cv::invertAffineTransform(transform_matrix, inverse_transform_matrix);
        const double* const tm_raw = inverse_transform_matrix.ptr<double>();
        const auto params = internal::warp_getWarpingAffineParameters(tm_raw, dstSize);

        return fk::Warping<fk::WarpType::Affine, std::decay_t<decltype(read)>>::build({ params, read });
    } else {
        const cv::Mat inverse_transform_matrix(transform_matrix.inv());
        const double* const tm_raw = inverse_transform_matrix.ptr<double>();
        const auto params = internal::warp_getWarpingPerspectiveParameters(tm_raw, dstSize);

        return fk::Warping<fk::WarpType::Perspective, std::decay_t<decltype(read)>>::build({ params, read });
    }
}

namespace internal {
    template <size_t BATCH>
    inline constexpr auto warp_batchAffineParameters_helper_rt(const std::array<cv::Mat, BATCH>& transform_matrices,
                                                               const std::array<cv::Size, BATCH>& dstSize, const size_t& idx) {
        cv::Mat inverse_transform_matrix;
        cv::invertAffineTransform(transform_matrices[idx], inverse_transform_matrix);
        const double* const tm_raw = inverse_transform_matrix.ptr<double>();
        return warp_getWarpingAffineParameters(tm_raw, dstSize[idx]);
    }

    template <size_t Idx, size_t BATCH>
    inline constexpr auto warp_batchAffineParameters_helper(const std::array<cv::Mat, BATCH>& transform_matrices,
                                                            const std::array<cv::Size, BATCH>& dstSize) {
        return warp_batchAffineParameters_helper_rt(transform_matrices, dstSize, Idx);
    }

    template <size_t BATCH>
    inline constexpr auto warp_batchPerspectiveParameters_helper_rt(const std::array<cv::Mat, BATCH>& transform_matrices,
                                                                    const std::array<cv::Size, BATCH>& dstSize,
                                                                    const size_t& idx) {
        const cv::Mat inverse_transform_matrix(transform_matrices[idx].inv());
        const double* const tm_raw = inverse_transform_matrix.ptr<double>();
        return warp_getWarpingPerspectiveParameters(tm_raw, dstSize[idx]);
    }

    template <size_t Idx, size_t BATCH>
    inline constexpr auto warp_batchPerspectiveParameters_helper(const std::array<cv::Mat, BATCH>& transform_matrices,
                                                                 const std::array<cv::Size, BATCH>& dstSize) {
        return warp_batchPerspectiveParameters_helper_rt(transform_matrices, dstSize, Idx);
    }

    template <enum fk::WarpType WT, size_t BATCH, size_t... Idx>
    inline constexpr auto warp_batchParameters_helper(const std::array<cv::Mat, BATCH>& transform_matrices,
                                                      const std::array<cv::Size, BATCH>& dstSize,
                                                      const std::index_sequence<Idx...>&) {
        if constexpr (WT == fk::WarpType::Affine) {
            return std::array{ warp_batchAffineParameters_helper<Idx>(transform_matrices, dstSize)... };
        } else {
            return std::array{ warp_batchPerspectiveParameters_helper<Idx>(transform_matrices, dstSize)... };
        }
    }

    template <enum fk::WarpType WT, size_t BATCH>
    inline constexpr auto warp_batchParameters_helper(const std::array<cv::Mat, BATCH>& transform_matrices,
                                                      const std::array<cv::Size, BATCH>& dstSize,
                                                      const int& usedPlanes) {
        std::array<fk::WarpingParameters<WT>, BATCH> temp;
        if constexpr (WT == fk::WarpType::Affine) {
            for (int i = 0; i < usedPlanes; ++i) {
                temp[i] = warp_batchAffineParameters_helper_rt(transform_matrices, dstSize, i);
            }
        } else {
            for (int i = 0; i < usedPlanes; ++i) {
                temp[i] = warp_batchPerspectiveParameters_helper_rt(transform_matrices, dstSize, i);
            }
        }
        return temp;
    }

    template <enum fk::WarpType WT, size_t BATCH>
    inline constexpr std::array<fk::WarpingParameters<WT>, BATCH> warp_batchParameters(const std::array<cv::Mat, BATCH>& transform_matrices,
                                                                                       const std::array<cv::Size, BATCH>& dstSize,
                                                                                       const int& usedPlanes = BATCH) {
        if (usedPlanes == static_cast<int>(BATCH)) {
            return warp_batchParameters_helper<WT>(transform_matrices, dstSize, std::make_index_sequence<BATCH>{});
        } else {
            return warp_batchParameters_helper<WT>(transform_matrices, dstSize, usedPlanes);
        }
    }
} // namespace internal

template <enum fk::WarpType WT, int InputType, size_t BATCH>
inline constexpr auto warp(const std::array<cv::cuda::GpuMat, BATCH>& inputs,
                           const std::array<cv::Mat, BATCH>& transform_matrices,
                           const std::array<cv::Size, BATCH>& dstSize) {
    for (int i = 0; i < BATCH; ++i) {
        if (InputType != inputs[i].type()) {
            throw std::runtime_error("Input type does not match the input type of the operation.");
        }
        if (transform_matrices[i].type() != CV_64FC1) {
            throw std::runtime_error("Transform matrix type should be CV_64FC1.");
        }
    }
    const auto fk_inputs = gpuMat2RawPtr2D_arr<CUDA_T(InputType)>(inputs);
    const auto readBatch = fk::PerThreadRead<fk::_2D, CUDA_T(InputType)>::build(fk_inputs);

    const auto fk_warpParams = internal::warp_batchParameters<WT>(transform_matrices, dstSize);

    const auto fk_batch_warp = fk::Warping<WT>::build(fk_warpParams);

    return readBatch.then(fk_batch_warp);
}

template <enum fk::WarpType WT, int InputType, size_t BATCH>
inline constexpr auto warp(const std::array<cv::cuda::GpuMat, BATCH>& inputs,
                           const std::array<cv::Mat, BATCH>& transform_matrices,
                           const cv::Size& dstSize) {
    return warp<WT, InputType>(inputs, transform_matrices, fk::make_set_std_array<BATCH>(dstSize));
}

template <enum fk::WarpType WT, int InputType, size_t BATCH>
inline constexpr auto warp(const std::array<cv::cuda::GpuMat, BATCH>& inputs,
                           const std::array<cv::Mat, BATCH>& transform_matrices,
                           const std::array<cv::Size, BATCH>& dstSize,
                           const int& usedPlanes, const cv::Scalar& defaultValue) {
    for (int i = 0; i < usedPlanes; ++i) {
        if (InputType != inputs[i].type()) {
            throw std::runtime_error("Input type does not match the input type of the operation.");
        }
        if (transform_matrices[i].type() != CV_64FC1) {
            throw std::runtime_error("Transform matrix type should be CV_64FC1.");
        }
    }
    const auto fk_inputs = gpuMat2RawPtr2D_arr<CUDA_T(InputType)>(inputs, usedPlanes);
    constexpr int DEFAULT_TYPE = CV_MAKETYPE(CV_32F, CV_MAT_CN(InputType));
    using DefaultType = CUDA_T(DEFAULT_TYPE);
    const auto fk_defaultValue = defaultValue == cv::Scalar() ? fk::make_set<DefaultType>(0.f) : cvScalar2CUDAV<DEFAULT_TYPE>::get(defaultValue);
    const auto readBatch = fk::PerThreadRead<fk::_2D, CUDA_T(InputType)>::build(fk_inputs);

    const auto fk_warpParams = internal::warp_batchParameters<WT>(transform_matrices, dstSize, usedPlanes);

    const auto fk_batch_warp = fk::Warping<WT>::build(usedPlanes, fk_defaultValue, fk_warpParams);

    return readBatch.then(fk_batch_warp);
}

template <enum fk::WarpType WT, int InputType, size_t BATCH>
inline constexpr auto warp(const std::array<cv::cuda::GpuMat, BATCH>& inputs,
                           const std::array<cv::Mat, BATCH>& transform_matrices,
                           const cv::Size& dstSize,
                           const int& usedPlanes, const cv::Scalar& defaultValue) {
    return warp<WT, InputType>(inputs, transform_matrices, fk::make_set_std_array<BATCH>(dstSize), usedPlanes, defaultValue);
}

template <typename BackIOp, int BATCH>
inline constexpr auto crop(const BackIOp& backIOp, const std::array<cv::Rect2d, BATCH>& rects) {
    return backIOp.then(crop(rects));
}

template <int O>
inline constexpr auto write(const cv::cuda::GpuMat& output) {
    return fk::Write<fk::PerThreadWrite<fk::_2D, CUDA_T(O)>>{ gpuMat2Ptr2D<CUDA_T(O)>(output).ptr() };
}

template <int O>
inline constexpr auto write(const cv::cuda::GpuMat& output, const cv::Size& plane) {
    return fk::Write<fk::PerThreadWrite<fk::_3D, CUDA_T(O)>>{ gpuMat2Tensor<CUDA_T(O)>(output, plane, 1).ptr() };
}

template <typename T>
inline constexpr auto write(const fk::Tensor<T>& output) {
    return fk::WriteInstantiableOperation<fk::PerThreadWrite<fk::_3D, T>>{ output };
}

template <bool ENABLE_THREAD_FUSION, typename... IOpTypes>
inline constexpr void executeOperations(const cv::cuda::Stream& stream, const IOpTypes&... instantiableOperations) {
    const cudaStream_t cu_stream = cv::cuda::StreamAccessor::getStream(stream);
    fk::executeOperations<ENABLE_THREAD_FUSION>(cu_stream, instantiableOperations...);
}

template <typename... IOpTypes>
inline constexpr void executeOperations(const cv::cuda::Stream& stream, const IOpTypes&... instantiableOperations) {
    executeOperations<true>(stream, instantiableOperations...);
}

template <bool ENABLE_THREAD_FUSION, typename... IOpTypes>
inline constexpr void executeOperations(const cv::cuda::GpuMat& input, const cv::cuda::Stream& stream,
                                        const IOpTypes&... instantiableOperations) {
    const cudaStream_t cu_stream = cv::cuda::StreamAccessor::getStream(stream);
    using InputType = fk::FirstInstantiableOperationInputType_t<IOpTypes...>;
    fk::executeOperations<ENABLE_THREAD_FUSION>(gpuMat2Ptr2D<InputType>(input), cu_stream, instantiableOperations...);
}

template <typename... IOpTypes>
inline constexpr void executeOperations(const cv::cuda::GpuMat& input, const cv::cuda::Stream& stream,
                                        const IOpTypes&... instantiableOperations) {
    executeOperations<true>(input, stream, instantiableOperations...);
}

template <bool ENABLE_THREAD_FUSION, typename... IOpTypes>
inline constexpr void executeOperations(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output,
                                        cv::cuda::Stream& stream, const IOpTypes&... instantiableOperations) {
    const cudaStream_t cu_stream = cv::cuda::StreamAccessor::getStream(stream);
    using InputType = fk::FirstInstantiableOperationInputType_t<IOpTypes...>;
    using OutputType = fk::LastInstantiableOperationOutputType_t<IOpTypes...>;
    fk::executeOperations<ENABLE_THREAD_FUSION>(gpuMat2Ptr2D<InputType>(input),
                                                gpuMat2Ptr2D<OutputType>(output), cu_stream, instantiableOperations...);
}

template <typename... IOpTypes>
inline constexpr void executeOperations(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output,
                                        cv::cuda::Stream& stream, const IOpTypes&... instantiableOperations) {
    executeOperations<true>(input, output, stream, instantiableOperations...);
}

// Batch reads
template <bool ENABLE_THREAD_FUSION, size_t Batch,  typename... IOpTypes>
inline constexpr void executeOperations(const std::array<cv::cuda::GpuMat, Batch>& input,
                                        const size_t& activeBatch, const cv::Scalar& defaultValue,
                                        const cv::cuda::Stream& stream,
                                        const IOpTypes&... instantiableOperations) {
    const cudaStream_t cu_stream = cv::cuda::StreamAccessor::getStream(stream);
    using InputType = fk::FirstInstantiableOperationInputType_t<IOpTypes...>;
    fk::executeOperations<ENABLE_THREAD_FUSION>(gpuMat2Ptr2D_arr<InputType, Batch>(input),
                                                                  activeBatch, cvScalar2CUDAV_t<InputType>::get(defaultValue),
                                                                  cu_stream, instantiableOperations...);
}

template <bool ENABLE_THREAD_FUSION, size_t Batch, typename... IOpTypes>
inline constexpr void executeOperations(const std::array<cv::cuda::GpuMat, Batch>& input,
                                        const cv::cuda::Stream& stream,
                                        const IOpTypes&... instantiableOperations) {
    const cudaStream_t cu_stream = cv::cuda::StreamAccessor::getStream(stream);
    using InputType = fk::FirstInstantiableOperationInputType_t<IOpTypes...>;
    fk::executeOperations<ENABLE_THREAD_FUSION>(gpuMat2Ptr2D_arr<InputType, Batch>(input),
                                                                  cu_stream, instantiableOperations...);
}

template <size_t Batch, typename... IOpTypes>
inline constexpr void executeOperations(const std::array<cv::cuda::GpuMat, Batch>& input,
                                        const size_t& activeBatch, const cv::Scalar& defaultValue,
                                        const cv::cuda::Stream& stream,
                                        const IOpTypes&... instantiableOperations) {
    executeOperations<true>(input, activeBatch, defaultValue, stream, instantiableOperations...);
}

template <size_t Batch, typename... IOpTypes>
inline constexpr void executeOperations(const std::array<cv::cuda::GpuMat, Batch>& input,
                                        const cv::cuda::Stream& stream,
                                        const IOpTypes&... instantiableOperations) {
    executeOperations<true>(input, stream, instantiableOperations...);
}

template <bool ENABLE_THREAD_FUSION, size_t Batch, typename... IOpTypes>
inline constexpr void executeOperations(const std::array<cv::cuda::GpuMat, Batch>& input,
                                        const size_t& activeBatch, const cv::Scalar& defaultValue,
                                        const cv::cuda::GpuMat& output, const cv::Size& outputPlane,
                                        const cv::cuda::Stream& stream,
                                        const IOpTypes&... instantiableOperations) {
    const cudaStream_t cu_stream = cv::cuda::StreamAccessor::getStream(stream);
    using InputType = fk::FirstInstantiableOperationInputType_t<IOpTypes...>;
    using OutputType = fk::LastInstantiableOperationOutputType_t<IOpTypes...>;
    fk::executeOperations<ENABLE_THREAD_FUSION>(gpuMat2Ptr2D_arr<InputType>(input),
                                                activeBatch, cvScalar2CUDAV_t<InputType>::get(defaultValue),
                                                gpuMat2Tensor<OutputType>(output, outputPlane, 1),
                                                cu_stream, instantiableOperations...);
}

template <bool ENABLE_THREAD_FUSION, size_t Batch, typename... IOpTypes>
inline constexpr void executeOperations(const std::array<cv::cuda::GpuMat, Batch>& input,
                                        const cv::cuda::GpuMat& output, const cv::Size& outputPlane,
                                        const cv::cuda::Stream& stream,
                                        const IOpTypes&... instantiableOperations) {
    const cudaStream_t cu_stream = cv::cuda::StreamAccessor::getStream(stream);
    using InputType = fk::FirstInstantiableOperationInputType_t<IOpTypes...>;
    using OutputType = fk::LastInstantiableOperationOutputType_t<IOpTypes...>;
    fk::executeOperations<ENABLE_THREAD_FUSION>(gpuMat2Ptr2D_arr<InputType>(input),
                                                gpuMat2Tensor<OutputType>(output, outputPlane, 1),
                                                cu_stream, instantiableOperations...);
}

template <size_t Batch, typename... IOpTypes>
inline constexpr void executeOperations(const std::array<cv::cuda::GpuMat, Batch>& input, 
                                        const size_t& activeBatch, const cv::Scalar& defaultValue,
                                        const cv::cuda::GpuMat& output, const cv::Size& outputPlane,
                                        const cv::cuda::Stream& stream, const IOpTypes&... instantiableOperations) {
    executeOperations<true>(input, activeBatch, defaultValue, output, outputPlane, stream, instantiableOperations...);
}

template <size_t Batch, typename... IOpTypes>
inline constexpr void executeOperations(const std::array<cv::cuda::GpuMat, Batch>& input,
                                        const cv::cuda::GpuMat& output, const cv::Size& outputPlane,
                                        const cv::cuda::Stream& stream, const IOpTypes&... instantiableOperations) {
    executeOperations<true>(input, output, outputPlane, stream, instantiableOperations...);
}

/* Copyright 2023 Mediaproduccion S.L.U. (Oscar Amoros Huguet)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

template <int I, int O, int COLOR_PLANES, int BATCH, fk::CircularTensorOrder CT_ORDER, fk::ColorPlanes CP_MODE = fk::ColorPlanes::Standard>
class CircularTensor : public fk::CircularTensor<CUDA_T(O), COLOR_PLANES, BATCH, CT_ORDER, CP_MODE> {
public:
    inline constexpr CircularTensor() {};

    inline constexpr CircularTensor(const uint& width_, const uint& height_, const int& deviceID_ = 0) :
        fk::CircularTensor<CUDA_T(O), COLOR_PLANES, BATCH, CT_ORDER, CP_MODE>(width_, height_, deviceID_) {};

    inline constexpr void Alloc(const uint& width_, const uint& height_, const int& deviceID_ = 0) {
        fk::CircularTensor<CUDA_T(O), COLOR_PLANES, BATCH, CT_ORDER, CP_MODE>::Alloc(width_, height_, deviceID_);
    }

    template <typename... IOpTypes>
    inline constexpr void update(const cv::cuda::Stream& stream, const cv::cuda::GpuMat& input, const IOpTypes&... instantiableOperationInstances) {
        const fk::Read<fk::PerThreadRead<fk::_2D, CUDA_T(I)>> readInstantiableOperation{
            {{(CUDA_T(I)*)input.data, { static_cast<uint>(input.cols), static_cast<uint>(input.rows), static_cast<uint>(input.step) }}}};
        fk::CircularTensor<CUDA_T(O), COLOR_PLANES, BATCH, CT_ORDER, CP_MODE>::update(cv::cuda::StreamAccessor::getStream(stream), readInstantiableOperation, instantiableOperationInstances...);
    }

    template <typename... IOpTypes>
    inline constexpr void update(const cv::cuda::Stream& stream, const IOpTypes&... instantiableOperationInstances) {
        fk::CircularTensor<CUDA_T(O), COLOR_PLANES, BATCH, CT_ORDER, CP_MODE>::update(cv::cuda::StreamAccessor::getStream(stream), instantiableOperationInstances...);
    }

    inline constexpr CUDA_T(O)* data() {
        return this->ptr_a.data;
    }
};
} // namespace cvGS
