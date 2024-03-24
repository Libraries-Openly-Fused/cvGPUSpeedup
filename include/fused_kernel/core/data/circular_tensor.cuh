/* Copyright 2023 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.*/

#pragma once

#include <fused_kernel/core/data/ptr_nd.cuh>
#include <fused_kernel/core/execution_model/grid_patterns.cuh>

namespace fk {

    enum CircularTensorOrder { NewestFirst, OldestFirst };

    template <CircularTensorOrder CTO, int BATCH>
    struct SequenceSelectorType {
        FK_HOST_DEVICE_FUSE uint at(const uint& index) {
            if constexpr (CTO == NewestFirst) {
                return index > 0 ? 2u : 1u;
            } else {
                return index != BATCH - 1 ? 2u : 1u;
            }
        }
    };

    template <CircularTensorOrder CT_ORDER>
    struct CTReadDirection;

    template <>
    struct CTReadDirection<CircularTensorOrder::NewestFirst> {
        static const CircularDirection dir{ Descendent };
    };

    template <>
    struct CTReadDirection<CircularTensorOrder::OldestFirst> {
        static const CircularDirection dir{ Ascendent };
    };

    template <CircularTensorOrder CT_ORDER>
    static constexpr CircularDirection CTReadDirection_v = CTReadDirection<CT_ORDER>::dir;

    enum ColorPlanes { Standard, Transposed };

    template <typename T, ColorPlanes CP_MODE>
    struct CoreType;

    template <typename T>
    struct CoreType<T, ColorPlanes::Standard> {
        using type = Tensor<T>;
    };

    template <typename T>
    struct CoreType<T, ColorPlanes::Transposed> {
        using type = TensorT<T>;
    };

    template <typename T, ColorPlanes CP_MODE>
    using CoreType_t = typename CoreType<T, CP_MODE>::type;

    template <typename T, int COLOR_PLANES, typename Enabler = void>
    struct CircularTensorStoreType {};

    template <typename T, int COLOR_PLANES>
    struct CircularTensorStoreType<T, COLOR_PLANES, std::enable_if_t<std::is_aggregate_v<T>&& COLOR_PLANES == 1>> {
        using type = T;
    };

    template <typename T, int COLOR_PLANES>
    struct CircularTensorStoreType<T, COLOR_PLANES, std::enable_if_t<!std::is_aggregate_v<T> && (COLOR_PLANES > 1)>> {
        using type = VectorType_t<T, COLOR_PLANES>;
    };

    template <typename T, int COLOR_PLANES, int BATCH, CircularTensorOrder CT_ORDER, ColorPlanes CP_MODE>
    class CircularTensor : public CoreType_t<T, CP_MODE> {

        using ParentType = CoreType_t<T, CP_MODE>;

        using StoreT = typename CircularTensorStoreType<T, COLOR_PLANES>::type;

        using WriteDeviceFunctions = TypeList<Write<TensorWrite<StoreT>>,
            Write<TensorSplit<StoreT>>,
            Write<TensorTSplit<StoreT>>>;

        using ReadDeviceFunctions = TypeList<Read<CircularTensorRead<CTReadDirection_v<CT_ORDER>, TensorRead<StoreT>, BATCH>>,
            Read<CircularTensorRead<CTReadDirection_v<CT_ORDER>, TensorPack<StoreT>, BATCH>>,
            Read<CircularTensorRead<CTReadDirection_v<CT_ORDER>, TensorTPack<StoreT>, BATCH>>>;

    public:
        __host__ inline constexpr CircularTensor() {};

        __host__ inline constexpr CircularTensor(const uint& width_, const uint& height_, const int& deviceID_ = 0) :
            ParentType(width_, height_, BATCH, COLOR_PLANES, MemType::Device, deviceID_),
            m_tempTensor(width_, height_, BATCH, COLOR_PLANES, MemType::Device, deviceID_) {};

        __host__ inline constexpr void Alloc(const uint& width_, const uint& height_, const int& deviceID_ = 0) {
            this->allocTensor(width_, height_, BATCH, COLOR_PLANES, MemType::Device, deviceID_);
            m_tempTensor.allocTensor(width_, height_, BATCH, COLOR_PLANES, MemType::Device, deviceID_);
        }

        template <typename... DeviceFunctionTypes>
        __host__ inline constexpr void update(const cudaStream_t& stream,
            const DeviceFunctionTypes&... deviceFunctionInstances) {
            const auto writeDeviceFunction = ppLast(deviceFunctionInstances...);
            using writeDFType = std::decay_t<decltype(writeDeviceFunction)>;
            using writeOpType = typename writeDFType::Operation;
            if constexpr (CP_MODE == ColorPlanes::Transposed) {
                static_assert(std::is_same_v<writeDFType, Write<TensorTSplit<StoreT>>>,
                    "Need to use TensorTSplitWrite as write function because you are using a transposed CircularTensor (CP_MODE = Transposed)");
            }
            using equivalentReadDFType = EquivalentType_t<writeDFType, WriteDeviceFunctions, ReadDeviceFunctions>;

            MidWrite<CircularTensorWrite<CircularDirection::Ascendent, writeOpType, BATCH>> updateWriteToTemp;
            updateWriteToTemp.params.first = m_nextUpdateIdx;
            updateWriteToTemp.params.params = m_tempTensor.ptr();

            const auto updateOps = buildOperationSequence_tup(insert_before_last(updateWriteToTemp, deviceFunctionInstances...));

            // Build copy pipeline
            equivalentReadDFType nonUpdateRead;
            nonUpdateRead.params.first = m_nextUpdateIdx;
            nonUpdateRead.params.params = m_tempTensor.ptr();
            nonUpdateRead.activeThreads.x = this->ptr_a.dims.width;
            nonUpdateRead.activeThreads.y = this->ptr_a.dims.height;
            nonUpdateRead.activeThreads.z = BATCH;

            const auto copyOps = buildOperationSequence(nonUpdateRead, writeDeviceFunction);

            dim3 grid((uint)ceil((float)this->ptr_a.dims.width / (float)this->adjusted_blockSize.x),
                (uint)ceil((float)this->ptr_a.dims.height / (float)this->adjusted_blockSize.y),
                BATCH);

            cuda_transform_divergent_batch<SequenceSelectorType<CT_ORDER, BATCH>> << <grid, this->adjusted_blockSize, 0, stream >> > (updateOps, copyOps);

            m_nextUpdateIdx = (m_nextUpdateIdx + 1) % BATCH;
            gpuErrchk(cudaGetLastError());
        }

    private:
        CoreType_t<T, CP_MODE> m_tempTensor;
        int m_nextUpdateIdx{ 0 };
    };
} // namespace fk
