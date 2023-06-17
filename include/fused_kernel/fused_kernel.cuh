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
#include "parameter_pack_utils.cuh"

namespace fk { // namespace FusedKernel
// generic operation structs
template <typename Operation>
struct ReadDeviceFunction {
    typename Operation::ParamsType params;
    dim3 activeThreads;
    using Op = Operation;
};

template <typename Operation>
struct BinaryDeviceFunction {
    typename Operation::ParamsType params;
    using Op = Operation;
};

template <typename Operation>
struct UnaryDeviceFunction {
    using Op = Operation;
};

template <typename Operation>
struct MidWriteDeviceFunction {
    typename Operation::ParamsType params;
    using Op = Operation;
};

template <typename Operation>
struct WriteDeviceFunction {
    typename Operation::ParamsType params;
    using Op = Operation;
};

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
    const auto writePattern = last(ops...);
    using WriteOperation = typename decltype(writePattern)::Op;

    const cg::thread_block g = cg::this_thread_block();

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

template <typename OpSelector, int BATCH, int OpSequenceNumber, typename ReadOperation, typename... Operations>
__device__ __forceinline__ constexpr void divergent_operate(const uint& z, const OperationSequence<ReadDeviceFunction<ReadOperation>, Operations...>& opSeq) {
    // If the threads with this z, arrived here, we assume they have to execute this operation sequence
    fk::apply(cuda_transform_d<ReadOperation, Operations...>, opSeq.args);
}

template <typename OpSelector, int BATCH, int OpSequenceNumber, typename ReadOperation, typename... Operations, typename... OperationSequences>
__device__ __forceinline__ constexpr void divergent_operate(const uint& z, const OperationSequence<ReadDeviceFunction<ReadOperation>, Operations...>& opSeq, const OperationSequences&... opSeqs) {
    if (OpSequenceNumber == OpSelector::at(z)) {
        fk::apply(cuda_transform_d<ReadOperation, Operations...>, opSeq.args);
    }
    else {
        divergent_operate<OpSelector, BATCH, OpSequenceNumber + 1>(z, opSeqs...);
    }
}

template <typename OpSelector, int BATCH, typename... OperationSequences>
__global__ void cuda_transform_divergent_batch(const OperationSequences... opSeqs) {
    const cg::thread_block g = cg::this_thread_block();
    const uint z = g.group_index().z;
    divergent_operate<OpSelector, BATCH, 1>(z, opSeqs...);
}

/* Copyright 2023 Mediaproduccion S.L.U. (Oscar Amoros Huguet)
   Copyright 2023 Mediaproduccion S.L.U. (David del Rio Astorga)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

template <int BATCH, int OpSequenceNumber, typename ReadOperation, typename... Operations>
__device__ __forceinline__ constexpr void divergent_operate(const uint& z, const Array<int, BATCH>& opSeqSelector, const OperationSequence<ReadDeviceFunction<ReadOperation>, Operations...>& opSeq) {
    // If the threads with this z, arrived here, we assume they have to execute this operation sequence
    fk::apply(cuda_transform_d<ReadOperation, Operations...>, opSeq.args);
}

template <int BATCH, int OpSequenceNumber, typename ReadOperation, typename... Operations, typename... OperationSequences>
__device__ __forceinline__ constexpr void divergent_operate(const uint& z, const Array<int, BATCH>& opSeqSelector, const OperationSequence<ReadDeviceFunction<ReadOperation>, Operations...>& opSeq, const OperationSequences&... opSeqs) {
    if (OpSequenceNumber == opSeqSelector.at[z]) {
        fk::apply(cuda_transform_d<ReadOperation, Operations...>, opSeq.args);
    } else {
        divergent_operate<BATCH, OpSequenceNumber + 1>(z, opSeqSelector, opSeqs...);
    }
}

template <int BATCH, typename... OperationSequences>
__global__ void cuda_transform_divergent_batch(const Array<int, BATCH> opSeqSelector, const OperationSequences... opSeqs) {
    const cg::thread_block g = cg::this_thread_block();
    const uint z = g.group_index().z;
    divergent_operate<BATCH, 1>(z, opSeqSelector, opSeqs...);
}
} // namespace FusedKernel
