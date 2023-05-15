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
#include "cuda_vector_utils.cuh"
#include "../external/opencv/modules/core/include/opencv2/core/cuda/vec_math.hpp"

#define DECL_TYPES_UNARY(I, O) using InputType = I; using OutputType = O;
#define DECL_TYPES_BINARY(I1, I2, O)  using InputType = I1; using ParamsType = I2; using OutputType = O;

namespace fk {

template <typename I1, typename I2=I1, typename O=I1>
struct BinarySum {
    FK_HOST_DEVICE_FUSE O exec(const I1& input_1, const I2& input_2) {return input_1 + input_2;}
    DECL_TYPES_BINARY(I1, I2, O)
};

template <typename I1, typename I2=I1, typename O=I1>
struct BinarySub {
    FK_HOST_DEVICE_FUSE O exec(const I1& input_1, const I2& input_2) {return input_1 - input_2;}
    DECL_TYPES_BINARY(I1, I2, O)
};

template <typename I1, typename I2=I1, typename O=I1>
struct BinaryMul {
    FK_HOST_DEVICE_FUSE O exec(const I1& input_1, const I2& input_2) { return input_1 * input_2; }
    DECL_TYPES_BINARY(I1, I2, O)
};

template <typename I1, typename I2=I1, typename O=I1>
struct BinaryDiv {
    FK_HOST_DEVICE_FUSE O exec(const I1& input_1, const I2& input_2) {return input_1 / input_2;}
    DECL_TYPES_BINARY(I1, I2, O)
};

template <typename T, int... idxs>
struct UnaryVectorReorder {
    FK_HOST_DEVICE_FUSE T exec(const T& input) {
        static_assert(validCUDAVec<T>, "Non valid CUDA vetor type: UnaryVectorReorder");
        static_assert(Channels<T>() >= 2, "Minimum number of channels is 2: UnaryVectorReorder");
        return VReorder<idxs...>::exec(input);
    }
    using idxType = typename VectorType<int,Channels<T>()>::type;
    DECL_TYPES_BINARY(T, idxType, T)
};

template <typename T>
struct BinaryVectorReorder {
    FK_HOST_DEVICE_FUSE T exec(const T& input, const typename VectorType<int,Channels<T>()>::type& idx) {
        static_assert(validCUDAVec<T>, "Non valid CUDA vetor type: UnaryVectorReorder");
        static_assert(Channels<T>() >= 2, "Minimum number of channels is 2: UnaryVectorReorder");
        using baseType = typename VectorTraits<T>::base;
        baseType* temp = (baseType*)&input; 
        if constexpr (Channels<T>() == 2) {
            return {temp[idx.x], temp[idx.y]};
        } else if constexpr (Channels<T>() == 3) {
            return {temp[idx.x], temp[idx.y], temp[idx.z]};
        } else {
            return {temp[idx.x], temp[idx.y], temp[idx.z], temp[idx.w]};
        }
    }
    using idxType = typename VectorType<int,Channels<T>()>::type;
    DECL_TYPES_BINARY(T, idxType, T)
};

template <typename I, typename O>
struct UnaryCast {
    FK_HOST_DEVICE_FUSE O exec(const I& input) { return saturate_cast<O>(input); }
    DECL_TYPES_UNARY(I, O)
};

}

#undef DECL_TYPES_UNARY
#undef DECL_TYPES_BINARY