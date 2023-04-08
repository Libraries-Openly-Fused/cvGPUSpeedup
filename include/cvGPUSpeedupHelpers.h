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

#include <fast_kernel/cuda_vector_utils.h>
#include <fast_kernel/memory_operation_types.h>
#include <cv2cuda_types.h>

#include <opencv2/core/cuda.hpp>

namespace cvGS {

namespace internal {

template <int I, typename Operator, typename Enabler = void>
struct split_t {};

template <int I, typename Operator>
struct split_t<I, Operator, std::enable_if_t<CV_MAT_CN(I) == 2>> {
    inline constexpr Operator operator()(std::vector<fk::Ptr_2D<BASE_CUDA_T(I)>>& output) {
        return { output.at(0), output.at(1) };
    }
};

template <int I, typename Operator>
struct split_t<I, Operator, std::enable_if_t<CV_MAT_CN(I) == 3>> {
    inline constexpr Operator operator()(std::vector<fk::Ptr_2D<BASE_CUDA_T(I)>>& output) {
        return { output.at(0), output.at(1), output.at(2) };
    }
};

template <int I, typename Operator>
struct split_t<I, Operator, std::enable_if_t<CV_MAT_CN(I) == 4>> {
    inline constexpr Operator operator()(std::vector<fk::Ptr_2D<BASE_CUDA_T(I)>>& output) {
        return { output.at(0), output.at(1), output.at(2), output.at(3) };
    }
};

template <int I, typename Operator, typename Enabler = void>
struct operate_t {};

template <int I, typename Operator>
struct operate_t<I, Operator, std::enable_if_t<CV_MAT_CN(I) == 1>> {
    inline constexpr Operator operator()(cv::Scalar& val) {
        return { static_cast<BASE_CUDA_T(I)>(val[0]) };
    }
};

template <int I, typename Operator>
struct operate_t<I, Operator, std::enable_if_t<CV_MAT_CN(I) == 2>> {
    inline constexpr Operator operator()(cv::Scalar& val) {
        return { fk::make_<CUDA_T(I)>(val[0], val[1]) };
    }
};

template <int I, typename Operator>
struct operate_t<I, Operator, std::enable_if_t<CV_MAT_CN(I) == 3>> {
    inline constexpr Operator operator()(cv::Scalar& val) {
        return { fk::make_<CUDA_T(I)>(val[0], val[1], val[2]) };
    }
};

template <int I, typename Operator>
struct operate_t<I, Operator, std::enable_if_t<CV_MAT_CN(I) == 4>> {
    inline constexpr Operator operator()(cv::Scalar& val) {
        return { fk::make_<CUDA_T(I)>(val[0], val[1], val[2], val[3]) };
    }
};
}
}
