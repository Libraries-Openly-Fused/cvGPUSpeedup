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

#include "operations.cuh"
#include "memory_operations.cuh"

namespace fk { // namespace FusedKernel
// generic operation struct

template <typename Operation>
struct ReadDeviceFunction {
    typename Operation::ParamsType params;
    dim3 activeThreads;
};

template <typename Operation>
struct BinaryDeviceFunction {
    typename Operation::ParamsType params;
};

template <typename Operation>
struct UnaryDeviceFunction {};

template <typename Operation>
struct MidWriteDeviceFunction {
    typename Operation::ParamsType params;
};

template <typename Operation>
struct WriteDeviceFunction {
    typename Operation::ParamsType params;
    using Op = Operation;
};

// Util to get the last parameter of a parameter pack
template <typename T>
__device__ __forceinline__ constexpr T last(const T& t) {
    return t;
}
template <typename T, typename... Args>
__device__ __forceinline__ constexpr auto last(const T& t, const Args&... args) {
    return last(args...);
}

// Recursive operate function
template <typename T>
__device__ __forceinline__ constexpr void operate(const Point& thread, const T& i_data) {
    return i_data;
}

template <typename Operation, typename... operations>
__device__ __forceinline__ constexpr auto operate(const Point& thread, const typename Operation::InputType& i_data, const BinaryDeviceFunction<Operation>& op, const operations&... ops) {
    return operate(thread, Operation::exec(i_data, op.params), ops...);
}

template <typename Operation, typename... operations>
__device__ __forceinline__ constexpr auto operate(const Point& thread, const typename Operation::InputType& i_data, const UnaryDeviceFunction<Operation>& op, const operations&... ops) {
    return operate(thread, Operation::exec(i_data), ops...);
}

template <typename Operation, typename... operations>
__device__ __forceinline__ constexpr auto operate(const Point& thread, const typename Operation::Type& i_data, const MidWriteDeviceFunction<Operation>& op, const operations&... ops) {
    Operation::exec(thread, i_data, op.params);
    return operate(thread, i_data, ops...);
}

template <typename Operation>
__device__ __forceinline__ constexpr typename Operation::Type operate(const Point& thread, const typename Operation::Type& i_data, const WriteDeviceFunction<Operation>& op) {
    return i_data;
}

template <typename ReadOperation, typename... operations>
__device__ __forceinline__ constexpr void cuda_transform_d(const ReadDeviceFunction<ReadOperation>& readPattern, const operations&... ops) {
    auto writePattern = last(ops...);
    using WriteOperation = typename decltype(writePattern)::Op;

    cg::thread_block g = cg::this_thread_block();

    const uint x = (g.dim_threads().x * g.group_index().x) + g.thread_index().x;
    const uint y = (g.dim_threads().y * g.group_index().y) + g.thread_index().y;
    const uint z =  g.group_index().z; // So far we only consider the option of using the z dimension to specify n (x*y) thread planes
    const Point thread{x, y, z};

    if (x < readPattern.activeThreads.x && y < readPattern.activeThreads.y && z < readPattern.activeThreads.z) {
        const auto tempI = ReadOperation::exec(thread, readPattern.params);
        if constexpr (sizeof...(ops) > 1) {
            const auto tempO = operate(thread, tempI, ops...);
            WriteOperation::exec(thread, tempO, writePattern.params);
        } else {
            WriteOperation::exec(thread, tempI, writePattern.params);
        }
    }
}

template <typename... operations>
__global__ void cuda_transform(const operations... ops) {
    cuda_transform_d(ops...);
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

template <typename... Args>
struct OperationSequence {
    std::tuple<const Args...> args;
};

template <typename... operations>
inline constexpr auto buildOperationSequence(const operations&... ops) {
    return OperationSequence<operations...> {{ops...}};
}

template <int BATCH, int OpSequenceNumber, typename Operations>
__device__ __forceinline__ constexpr void divergent_operate(const uint& z, const int (&opSeqSelector)[BATCH], const OperationSequence<Operations>& opSeq) {
    // If the threads with this z, arrived here, we assume they have to execute this operation sequence
    std::apply(cuda_transform_d, opSeq.args);
}

template <int BATCH, int OpSequenceNumber, typename Operations, typename... OperationSequences>
__device__ __forceinline__ constexpr void divergent_operate(const uint& z, const int (&opSeqSelector)[BATCH], const OperationSequence<Operations>& opSeq, const OperationSequences&... opSeqs) {
    if (OpSequenceNumber == opSeqSelector[z]) {
        std::apply(cuda_transform_d, opSeq.args);
    } else {
        divergent_operate<BATCH, OpSequenceNumber + 1>(z, opSeqSelector, opSeqs...);
    }
}

template <int BATCH, typename... OperationSequences>
__global__ void cuda_transform_divergent_batch(const int opSeqSelector[BATCH], const OperationSequences... opSeqs) {
    const cg::thread_block g = cg::this_thread_block();
    const uint z = g.group_index().z;
    divergent_operate<BATCH, 1>(z, opSeqSelector, opSeqs...);
}


} // namespace FusedKernel
