/* Copyright 2025 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef VERTICAL_FUSION_STATIC_LOOP_CUH
#define VERTICAL_FUSION_STATIC_LOOP_CUH

#include <cvGPUSpeedup.cuh>
#include <fused_kernel/algorithms/basic_ops/static_loop.cuh>

template <int CV_TYPE_I, int CV_TYPE_O, int OPS_PER_ITER, size_t NumOps, typename DeviceFunction>
struct VerticalFusion {
    template <size_t BATCH>
    static inline void execute(const std::array<cv::cuda::GpuMat, BATCH>& crops,
        const cv::cuda::Stream& cv_stream,
        const float& alpha,
        const cv::cuda::GpuMat& d_tensor_output,
        const cv::Size& cropSize,
        const DeviceFunction& dFunc) {
        using InputType = CUDA_T(CV_TYPE_I);
        using OutputType = CUDA_T(CV_TYPE_O);

        if constexpr (NumOps < INCREMENT) {
            using Loop = fk::Binary<fk::StaticLoop<typename DeviceFunction::Operation, NumOps / OPS_PER_ITER>>;

            const Loop loop{{dFunc.params}};

            cvGS::executeOperations<false>(crops, cv_stream, cvGS::convertTo<CV_TYPE_I, CV_TYPE_O>((float)alpha), loop, cvGS::write<CV_TYPE_O>(d_tensor_output, cropSize));
        } else {
            using Loop = fk::Binary<fk::StaticLoop<fk::StaticLoop<
            typename DeviceFunction::Operation, INCREMENT / OPS_PER_ITER>, NumOps / INCREMENT>>;

            const Loop loop{{dFunc.params}};

            cvGS::executeOperations<false>(crops, cv_stream, cvGS::convertTo<CV_TYPE_I, CV_TYPE_O>((float)alpha), loop, cvGS::write<CV_TYPE_O>(d_tensor_output, cropSize));
        }
        
    }
};

#endif
