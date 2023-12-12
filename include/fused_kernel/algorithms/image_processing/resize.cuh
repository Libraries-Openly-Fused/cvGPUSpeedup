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

#pragma once

#include <fused_kernel/core/external/carotene/saturate_cast.hpp>
#include <fused_kernel/core/execution_model/device_functions.cuh>
#include <fused_kernel/core/fusionable_operations/memory_operations.cuh>
#include <fused_kernel/algorithms/image_processing/interpolation.cuh>

namespace fk {

    /*template <typename PixelReadOp, InterpolationType INTER_T>
    struct ResizeRead {
        using InterpolationOp = Interpolate<PixelReadOp, INTER_T>;
        using OutputType = typename InterpolationOp::OutputType;
        using ParamsType = ResizeReadParams<InterpolationOp>;
        using InstanceType = ReadType;
        static __device__ __forceinline__ const OutputType exec(const Point& thread, const ParamsType& params) {
            // This is what makes the interpolation a resize operation
            const float src_x = thread.x * params.fx;
            const float src_y = thread.y * params.fy;

            static_assert(std::is_same_v<typename InterpolationOp::InputType, float2>, "Wrong InputType for interpolation operation.");
            return InterpolationOp::exec(make_<float2>(src_x, src_y), params.params);
        }
    };*/

    struct ComputeResizePoint {
        using OutputType = float2;
        using ParamsType = float2;
        using InstanceType = ReadType;
        static __device__ __forceinline__ const OutputType exec(const Point& thread, const ParamsType& params) {
            // This is what makes the interpolation a resize operation
            const float fx = params.x;
            const float fy = params.y;

            const float src_x = thread.x * fx;
            const float src_y = thread.y * fy;

            return { src_x, src_y };
        }
    };

    template <typename PixelReadOp, InterpolationType IType>
    using ResizeRead = ComposedOperationSequence<
                           ComputeResizePoint,
                           ComputeInterpolationPoints<IType>,
                           ReadInterpolationPoints<PixelReadOp, IType>,
                           InterpolateSlice<typename ReadInterpolationPoints<PixelReadOp, IType>::OutputType,
                                            VectorType_t<float, cn<typename ReadInterpolationPoints<PixelReadOp, IType>::InputType>>,
                                            IType>
                       >;

    enum AspectRatio { PRESERVE_AR = 0, IGNORE_AR = 1, PRESERVE_AR_RN_EVEN = 2 };

    template <typename PixelReadOp, InterpolationType IType>
    inline const auto resize(const typename PixelReadOp::ParamsType& input,
                             const Size& srcSize, const Size& dstSize) {
        const double cfx = static_cast<double>(dstSize.width) / srcSize.width;
        const double cfy = static_cast<double>(dstSize.height) / srcSize.height;

        Read<ResizeRead<PerThreadRead<_2D, uchar4>, InterpolationType::INTER_LINEAR>> resizeInstance{};

        Get<0>::params(resizeInstance.params) = {0.f,0.f};
        Get<1>::params(resizeInstance.params) = {0,0};
        Get<2>::params(resizeInstance.params) = input;

        using MyList = TypeList<float, int, char, char, char>;

        using MyNewList = InsertType_t<0, uint, MyList>;

        TypeAt_t<0, MyNewList>;

        return resizeInstance;
    }

    template <typename I, InterpolationType IType>
    inline const auto resize(const RawPtr<_2D, I>& input, const Size& dSize, const double& fx, const double& fy) {
        const fk::Size sourceSize(input.dims.width, input.dims.height);
        if (dSize.width != 0 && dSize.height != 0) {
            const double cfx = static_cast<double>(dSize.width) / input.dims.width;
            const double cfy = static_cast<double>(dSize.height) / input.dims.height;

            using ResizeType = Read<ResizeRead<PerThreadRead<_2D, I>, InterpolationType::INTER_LINEAR>>;

            ResizeType resizeInstance{};

            Get<0>::params(resizeInstance.params) = { static_cast<float>(cfx),
                                                      static_cast<float>(cfy) };
            Get<1>::params(resizeInstance.params) = dSize;
            Get<2>::params(resizeInstance.params) = input;

            constexpr bool areSame = std::is_same_v<FirstType, FirstType_>;

            return ResizeType{  {
                                  { static_cast<float>(cfx), static_cast<float>(cfy) },
                                  { dSize, { input } }
                                }
                              };
        } else {
            return ResizeType
            {   { {input}, static_cast<float>(1.0 / fx), static_cast<float>(1.0 / fy) },
                { CAROTENE_NS::internal::saturate_cast<uint>(input.dims.width * fx),
                  CAROTENE_NS::internal::saturate_cast<uint>(input.dims.height * fy) }
            };
        }
    }

    template <typename PixelReadOp, typename O, InterpolationType IType, int NPtr, AspectRatio AR>
    inline const auto resize(const std::array<typename PixelReadOp::ParamsType, NPtr>& input,
        const Size& dsize, const int& usedPlanes,
        const O& backgroundValue = fk::make_set<O>(0)) {
        using ResizeArrayIgnoreType = Read<BatchRead<ResizeRead<PixelReadOp, IType>, NPtr>>;
        using ResizeArrayPreserveType = Read<BatchRead<ApplyROI<ResizeRead<PixelReadOp, IType>, OFFSET_THREADS>, NPtr>>;
        using ResizeArrayPreserveRoundEvenType = Read<BatchRead<ApplyROI<ResizeRead<PixelReadOp, IType>, OFFSET_THREADS>, NPtr>>;
        using ResizeArrayType = TypeAt_t<AR, TypeList<ResizeArrayPreserveType, ResizeArrayIgnoreType, ResizeArrayPreserveRoundEvenType>>;

        ResizeArrayType resizeArray;
        // dsize is the size of the destination pointer, for each image
        resizeArray.activeThreads.x = dsize.width;
        resizeArray.activeThreads.y = dsize.height;
        resizeArray.activeThreads.z = usedPlanes;

        for (int i = 0; i < usedPlanes; i++) {
            const fk::PtrDims<fk::_2D> dims = input[i].dims;

            // targetWidth and targetHeight are the dimensions for the resized image
            int targetWidth, targetHeight;
            fk::ResizeReadParams<Interpolate<PixelReadOp, IType>>* interParams;
            if constexpr (AR != IGNORE_AR) {
                float scaleFactor = dsize.height / (float)dims.height;
                targetHeight = dsize.height;
                targetWidth = static_cast<int> (round(scaleFactor * dims.width));
                if constexpr (AR == PRESERVE_AR_RN_EVEN) {
                    // We round to the next even integer smaller or equal to targetWidth
                    targetWidth -= targetWidth % 2;
                }
                if (targetWidth > dsize.width) {
                    scaleFactor = dsize.width / (float)dims.width;
                    targetWidth = dsize.width;
                    targetHeight = static_cast<int> (round(scaleFactor * dims.height));
                    if constexpr (AR == PRESERVE_AR_RN_EVEN) {
                        // We round to the next even integer smaller or equal to targetHeight
                        targetHeight -= targetHeight % 2;
                    }
                }
                resizeArray.activeThreads.z = NPtr;
                resizeArray.params[i].x1 = (dsize.width - targetWidth) / 2;
                resizeArray.params[i].x2 = resizeArray.params[i].x1 + targetWidth - 1;
                resizeArray.params[i].y1 = (dsize.height - targetHeight) / 2;
                resizeArray.params[i].y2 = resizeArray.params[i].y1 + targetHeight - 1;
                resizeArray.params[i].defaultValue = backgroundValue;
                interParams = &resizeArray.params[i].params;
            } else {
                targetWidth = dsize.width;
                targetHeight = dsize.height;
                interParams = &resizeArray.params[i];
            }
            interParams->params = { input[i] };
            interParams->fx = static_cast<float>(1.0 / (static_cast<double>(targetWidth) / (double)dims.width));
            interParams->fy = static_cast<float>(1.0 / (static_cast<double>(targetHeight) / (double)dims.height));
        }

        if constexpr (AR != IGNORE_AR) {
            for (int i = usedPlanes; i < NPtr; i++) {
                resizeArray.params[i].x1 = -1;
                resizeArray.params[i].x2 = -1;
                resizeArray.params[i].y1 = -1;
                resizeArray.params[i].y2 = -1;
                resizeArray.params[i].defaultValue = backgroundValue;
            }
        }
        return resizeArray;
    }

}; // namespace fk


