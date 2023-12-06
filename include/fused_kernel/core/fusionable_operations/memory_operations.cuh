/* 
   Copyright 2023 Oscar Amoros Huguet

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
#include <fused_kernel/core/utils/cuda_vector_utils.cuh>
#include <fused_kernel/core/fusionable_operations/operations.cuh>
#include <fused_kernel/algorithms/image_processing/color_conversion.cuh>

namespace fk {

    template <ND D, typename T>
    struct PerThreadRead {
        using InputType = Point;
        using OutputType = T;
        using ParamsType = RawPtr<D, T>;
        using InstanceType = ReadType;
        FK_DEVICE_FUSE OutputType exec(const InputType& thread, const ParamsType& ptr) {
            return *PtrAccessor<D>::cr_point(thread, ptr);
        }
    };

    template <ND D, typename T>
    struct PerThreadWrite {
        using InputType = T;
        using ParamsType = RawPtr<D, T>;
        using InstanceType = WriteType;
        FK_DEVICE_FUSE void exec(const Point& thread, const InputType& input, const ParamsType& output) {
            *PtrAccessor<D>::point(thread, output) = input;
        }
    };

    template <typename T>
    struct TensorRead {
        using OutputType = T;
        using ParamsType = RawPtr<_3D, T>;
        using InstanceType = ReadType;
        FK_DEVICE_FUSE OutputType exec(const Point& thread, const ParamsType& ptr) {
            return *PtrAccessor<_3D>::cr_point(thread, ptr);
        }
    };

    template <typename T>
    struct TensorWrite {
        using InputType = T;
        using ParamsType = RawPtr<_3D, T>;
        using InstanceType = WriteType;
        FK_DEVICE_FUSE void exec(const Point& thread, const InputType& input, const ParamsType& output) {
            *PtrAccessor<_3D>::point(thread, output) = input;
        }
    };


    template <typename T>
    struct TensorSplit {
        using InputType = T;
        using ParamsType = RawPtr<_3D, typename VectorTraits<T>::base>;
        using InstanceType = WriteType;
        FK_DEVICE_FUSE void exec(const Point& thread, const InputType& input, const ParamsType& ptr) {
            static_assert(cn<InputType> >= 2, "Wrong type for split tensor write. It must be one of <type>2, <type>3 or <type>4.");

            const int planePixels = ptr.dims.width * ptr.dims.height;

            using OutputType = typename VectorTraits<InputType>::base;
            OutputType* const work_plane = PtrAccessor<_3D>::point(thread, ptr);
            *work_plane = input.x;
            *(work_plane + planePixels) = input.y;
            if constexpr (cn<InputType> >= 3) {
                *(work_plane + (planePixels * 2)) = input.z;
            }
            if constexpr (cn<InputType> == 4) {
                *(work_plane + (planePixels * 3)) = input.w;
            }
        }
    };

    template <typename T>
    struct TensorTSplit {
        using InputType = T;
        using ParamsType = RawPtr<T3D, typename VectorTraits<T>::base>;
        using InstanceType = WriteType;
        FK_DEVICE_FUSE void exec(const Point& thread, const InputType& input, const ParamsType& ptr) {
            static_assert(cn<InputType> >= 2, "Wrong type for split tensor write. It must be one of <type>2, <type>3 or <type>4.");

            using OutputType = typename VectorTraits<InputType>::base;
            *PtrAccessor<T3D>::point(thread, ptr, 0) = input.x;
            *PtrAccessor<T3D>::point(thread, ptr, 1) = input.y;
            if constexpr (cn<InputType> >= 3) {
                *PtrAccessor<T3D>::point(thread, ptr, 2) = input.z;
            }
            if constexpr (cn<InputType> == 4) {
                *PtrAccessor<T3D>::point(thread, ptr, 3) = input.w;
            }
        }
    };

    template <typename T>
    struct TensorPack {
        using InputType = VBase<T>;
        using OutputType = T;
        using ParamsType = RawPtr<_3D, InputType>;
        using InstanceType = ReadType;
        FK_DEVICE_FUSE OutputType exec(const Point& thread, const ParamsType& ptr) {
            static_assert(cn<OutputType> >= 2, "Wrong type for split tensor read. It must be one of <type>2, <type>3 or <type>4.");

            const int planePixels = ptr.dims.width * ptr.dims.height;

            const InputType* const work_plane = PtrAccessor<_3D>::cr_point(thread, ptr);
            if constexpr (cn<OutputType> == 2) {
                return make_<OutputType>(*work_plane, *(work_plane + planePixels));
            } else if constexpr (cn<OutputType> == 3) {
                return make_<OutputType>(*work_plane, *(work_plane + planePixels),
                    *(work_plane + (planePixels * 2)));
            } else {
                return make_<OutputType>(*work_plane,
                    *(work_plane + planePixels),
                    *(work_plane + (planePixels * 2)),
                    *(work_plane + (planePixels * 3)));
            }
        }
    };

    template <typename T>
    struct TensorTPack {
        using InputType = VBase<T>;
        using OutputType = T;
        using ParamsType = RawPtr<T3D, InputType>;
        using InstanceType = ReadType;
        FK_DEVICE_FUSE OutputType exec(const Point& thread, const ParamsType& ptr) {
            static_assert(cn<OutputType> >= 2, "Wrong type for split tensor read. It must be one of <type>2, <type>3 or <type>4.");

            const InputType x = *PtrAccessor<T3D>::cr_point(thread, ptr, 0);
            if constexpr (cn<OutputType> == 2) {
                const InputType y = *PtrAccessor<T3D>::cr_point(thread, ptr, 1);
                return make_<OutputType>(x, y);
            } else if constexpr (cn<OutputType> == 3) {
                const InputType y = *PtrAccessor<T3D>::cr_point(thread, ptr, 1);
                const InputType z = *PtrAccessor<T3D>::cr_point(thread, ptr, 2);
                return make_<OutputType>(x, y, z);
            } else {
                const InputType y = *PtrAccessor<T3D>::cr_point(thread, ptr, 1);
                const InputType z = *PtrAccessor<T3D>::cr_point(thread, ptr, 2);
                const InputType w = *PtrAccessor<T3D>::cr_point(thread, ptr, 3);
                return make_<OutputType>(x, y, z, w);
            }
        }
    };

    template <ND D, typename T, typename Enabler = void>
    struct SplitWriteParams {};

    template <ND D, typename T>
    struct SplitWriteParams<D, T, typename std::enable_if_t<cn<T> == 2>> {
        RawPtr<D, decltype(T::x)> x;
        RawPtr<D, decltype(T::y)> y;
    };

    template <ND D, typename T>
    struct SplitWriteParams<D, T, typename std::enable_if_t<cn<T> == 3>> {
        RawPtr<D, decltype(T::x)> x;
        RawPtr<D, decltype(T::y)> y;
        RawPtr<D, decltype(T::z)> z;
    };

    template <ND D, typename T>
    struct SplitWriteParams<D, T, typename std::enable_if_t<cn<T> == 4>> {
        RawPtr<D, decltype(T::x)> x;
        RawPtr<D, decltype(T::y)> y;
        RawPtr<D, decltype(T::z)> z;
        RawPtr<D, decltype(T::w)> w;
    };

    template <ND D, typename T>
    struct SplitWrite {
        using InputType = T;
        using ParamsType = SplitWriteParams<D, T>;
        using InstanceType = WriteType;
        FK_DEVICE_FUSE void exec(const Point& thread, const InputType& input, const ParamsType& params) {
            static_assert(cn<InputType> >= 2, "Wrong type for split write. It must be one of <type>2, <type>3 or <type>4.");
            *PtrAccessor<D>::point(thread, params.x) = input.x;
            *PtrAccessor<D>::point(thread, params.y) = input.y;
            if constexpr (cn<InputType> >= 3) *PtrAccessor<D>::point(thread, params.z) = input.z;
            if constexpr (cn<InputType> == 4) *PtrAccessor<D>::point(thread, params.w) = input.w;
        }
    };

    template <typename Operation, int NPtr>
    struct BatchRead {
        using OutputType = typename Operation::OutputType;
        using ParamsType = typename Operation::ParamsType[NPtr];
        using InstanceType = ReadType;
        FK_DEVICE_FUSE const OutputType exec(const Point& thread, const typename Operation::ParamsType(&params)[NPtr]) {
            return Operation::exec(thread, params[thread.z]);
        }
    };

    template <typename Operation, int NPtr>
    struct BatchWrite {
        using InputType = typename Operation::InputType;
        using ParamsType = typename Operation::ParamsType[NPtr];
        using InstanceType = WriteType;
        FK_DEVICE_FUSE void exec(const Point& thread, const InputType& input, const typename Operation::ParamsType(&params)[NPtr]) {
            Operation::exec(thread, input, params[thread.z]);
        }
    };

    template <typename InperpolationOp>
    struct ResizeReadParams {
        typename InperpolationOp::ParamsType params;
        float fx;
        float fy;
    };

    /* The following code has the following copy right

       Copyright 2023 Mediaproduccion S.L.U. (Oscar Amoros Huget)
       Copyright 2023 Mediaproduccion S.L.U. (Guillermo Oyarzun Altamirano)

       Licensed under the Apache License, Version 2.0 (the "License");
       you may not use this file except in compliance with the License.
       You may obtain a copy of the License at

           http://www.apache.org/licenses/LICENSE-2.0

       Unless required by applicable law or agreed to in writing, software
       distributed under the License is distributed on an "AS IS" BASIS,
       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
       See the License for the specific language governing permissions and
       limitations under the License. */

    enum CircularDirection { Ascendent, Descendent };

    template <typename ParamsType>
    struct CircularMemoryParams {
        uint first;
        ParamsType params;
    };

    template <CircularDirection direction, int BATCH>
    FK_HOST_DEVICE_CNST Point computeCircularThreadIdx(const Point& currentIdx, const uint& fst) {
        if constexpr (direction == CircularDirection::Ascendent) {
            const uint z = currentIdx.z + fst;
            return { currentIdx.x, currentIdx.y, z >= BATCH ? z - BATCH : z };
        } else {
            const int z = fst - currentIdx.z;
            return { currentIdx.x, currentIdx.y, z < 0 ? static_cast<uint>(BATCH + z) : static_cast<uint>(z) };
        }
    }

    template <CircularDirection direction, typename Operation, int BATCH>
    struct CircularBatchRead {
        using OutputType = typename Operation::OutputType;
        using ParamsType = CircularMemoryParams<typename Operation::ParamsType[BATCH]>;
        using InstanceType = ReadType;
        FK_DEVICE_FUSE const OutputType exec(const Point& thread, const ParamsType& c_params) {
            const Point newThreadIdx = computeCircularThreadIdx<direction, BATCH>(thread, c_params.first);
            return Operation::exec(newThreadIdx, c_params.params[newThreadIdx.z]);
        }
    };

    template <CircularDirection direction, typename Operation, int BATCH>
    struct CircularBatchWrite {
        using InputType = typename Operation::InputType;
        using ParamsType = CircularMemoryParams<typename Operation::ParamsType[BATCH]>;
        using InstanceType = WriteType;
        FK_DEVICE_FUSE void exec(const Point& thread, const InputType& input, const ParamsType& c_params) {
            const Point newThreadIdx = computeCircularThreadIdx<direction, BATCH>(thread, c_params.first);
            Operation::exec(newThreadIdx, input, c_params.params[newThreadIdx.z]);
        }
    };

    template <CircularDirection direction, typename Operation, int BATCH>
    struct CircularTensorRead {
        using OutputType = typename Operation::OutputType;
        using ParamsType = CircularMemoryParams<typename Operation::ParamsType>;
        using InstanceType = ReadType;
        FK_DEVICE_FUSE const OutputType exec(const Point& thread, const ParamsType& c_params) {
            const Point newThreadIdx = computeCircularThreadIdx<direction, BATCH>(thread, c_params.first);
            return Operation::exec(newThreadIdx, c_params.params);
        }
    };

    template <CircularDirection direction, typename Operation, int BATCH>
    struct CircularTensorWrite {
        using InputType = typename Operation::InputType;
        using ParamsType = CircularMemoryParams<typename Operation::ParamsType>;
        using InstanceType = WriteType;
        FK_DEVICE_FUSE void exec(const Point& thread, const InputType& input, const ParamsType& c_params) {
            const Point newThreadIdx = computeCircularThreadIdx<direction, BATCH>(thread, c_params.first);
            Operation::exec(newThreadIdx, input, c_params.params);
        }
    };

    enum ROI { OFFSET_THREADS, KEEP_THREAD_IDX };

    template <typename Operation>
    struct ApplyROIParams {
        int x1, y1; // Top left
        int x2, y2; // Bottom right
        typename Operation::OutputType defaultValue{};
        typename Operation::ParamsType params;
    };

    template <typename Operation, ROI USE>
    struct ApplyROI {
        using OutputType = typename Operation::OutputType;
        using ParamsType = ApplyROIParams<Operation>;
        using InstanceType = ReadType;
        static __device__ __forceinline__ const OutputType exec(const Point& thread, const ParamsType& params) {
            if (thread.x >= params.x1 && thread.x <= params.x2 && thread.y >= params.y1 && thread.y <= params.y2) {
                if constexpr (USE == OFFSET_THREADS) {
                    const Point roiThread(thread.x - params.x1, thread.y - params.y1, thread.z);
                    return Operation::exec(roiThread, params.params);
                } else {
                    return Operation::exec(thread, params.params);
                }
            } else {
                return params.defaultValue;
            }
        }
    };
} //namespace fk
