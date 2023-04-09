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
#include "memory_operation_types.cuh"

namespace fk {

template <typename Operator, typename O>
struct memory_write_scalar_2D {
    MemPatterns<O> x;
    Operator nv_operator;
};

template <typename Operator, typename I, typename Enabler=void>
struct split_write_scalar_2D {};

template <typename Operator, typename I>
struct split_write_scalar_2D<Operator, I, typename std::enable_if_t<NUM_COMPONENTS(I) == 2>> {
    MemPatterns<decltype(I::x)> x;
    MemPatterns<decltype(I::y)> y;
    Operator nv_operator;
};

template <typename Operator, typename I>
struct split_write_scalar_2D<Operator, I, typename std::enable_if_t<NUM_COMPONENTS(I) == 3>> {
    MemPatterns<decltype(I::x)> x;
    MemPatterns<decltype(I::y)> y;
    MemPatterns<decltype(I::z)> z;
    Operator nv_operator;
};

template <typename Operator, typename I>
struct split_write_scalar_2D<Operator, I, typename std::enable_if_t<NUM_COMPONENTS(I) == 4>> {
    MemPatterns<decltype(I::x)> x;
    MemPatterns<decltype(I::y)> y;
    MemPatterns<decltype(I::z)> z;
    MemPatterns<decltype(I::w)> w;
    Operator nv_operator;
};

}