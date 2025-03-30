/* Copyright 2023-2025 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_INSTANTIABLE_DATA_PARALLEL_PATTERNS
#define FK_INSTANTIABLE_DATA_PARALLEL_PATTERNS

#include <cooperative_groups.h>

namespace cooperative_groups {};
namespace cg = cooperative_groups;

#include <fused_kernel/core/utils/parameter_pack_utils.cuh>
#include <fused_kernel/core/execution_model/instantiable_operations.cuh>
#include <fused_kernel/core/execution_model/thread_fusion.cuh>

namespace fk { // namespace FusedKernel
    template <bool THREAD_FUSION, typename... IOps>
    struct BuildTFI {
        using ReadOp = typename FirstType_t<IOps...>::Operation;
        using WriteOp = typename LastType_t<IOps...>::Operation;
        using TFI =
            ThreadFusionInfo<typename ReadOp::ReadDataType,
                             typename WriteOp::WriteDataType,
                             isThreadFusionEnabled<THREAD_FUSION, IOps...>()>;
    };

    template <typename Enabler, bool THREAD_FUSION, typename... IOps>
    struct TransformDPPDetails_;
    
    template <bool THREAD_FUSION, typename... IOps>
    struct TransformDPPDetails_<std::enable_if_t<BuildTFI<THREAD_FUSION, IOps...>::TFI::ENABLED, void>,
                               THREAD_FUSION, IOps...> {
        ActiveThreads activeThreads;
        bool threadDivisible;
        using TFI = typename BuildTFI<THREAD_FUSION, IOps...>::TFI;
    };

    template <bool THREAD_FUSION, typename... IOps>
    struct TransformDPPDetails_<std::enable_if_t<!BuildTFI<THREAD_FUSION, IOps...>::TFI::ENABLED, void>,
                                THREAD_FUSION, IOps...> {
        using TFI = typename BuildTFI<THREAD_FUSION, IOps...>::TFI;
    };

    template <bool THREAD_FUSION, typename... IOps>
    using TransformDPPDetails = TransformDPPDetails_<void, THREAD_FUSION, IOps...>;

    template <typename DPPDetails = void, bool THREAD_DIVISIBLE = true>
    struct TransformDPP {
        using Details = DPPDetails;

        template <typename T, typename IOp, typename... IOpTypes>
        FK_DEVICE_FUSE auto operate(const Point& thread, const T& i_data, const IOp& iOp, const IOpTypes&... iOpInstances) {
            if constexpr (IOp::template is<WriteType>) {
                return i_data;
            // MidWriteOperation with continuations, based on FusedOperation
            } else if constexpr (IOp::template is<MidWriteType> && isMidWriteType<typename IOp::Operation>) {
                return IOp::Operation::exec(thread, i_data, iOp);
            } else if constexpr (IOp::template is<MidWriteType> && !isMidWriteType<typename IOp::Operation>) {
                IOp::Operation::exec(thread, i_data, iOp);
                return i_data;
            } else {
                return operate(thread, compute(i_data, iOp), iOpInstances...);
            }
        }

        template <uint IDX, typename TFI, typename InputType, typename... IOpTypes>
        FK_DEVICE_FUSE auto operate_idx(const Point& thread, const InputType& input, const IOpTypes&... instantiableOperationInstances) {
            return operate(thread, TFI::get<IDX>(input), instantiableOperationInstances...);
        }

        template <typename TFI, typename InputType, uint... IDX, typename... IOpTypes>
        FK_DEVICE_FUSE auto operate_thread_fusion_impl(std::integer_sequence<uint, IDX...> idx, const Point& thread,
                                                        const InputType& input, const IOpTypes&... instantiableOperationInstances) {
            return TFI::make(operate_idx<IDX, TFI>(thread, input, instantiableOperationInstances...)...);
        }

        template <typename TFI, typename InputType, typename... IOpTypes>
        FK_DEVICE_FUSE auto operate_thread_fusion(const Point& thread, const InputType& input, const IOpTypes&... instantiableOperationInstances) {
            if constexpr (TFI::elems_per_thread == 1) {
                return operate(thread, input, instantiableOperationInstances...);
            } else {
                return operate_thread_fusion_impl<TFI>(std::make_integer_sequence<uint, TFI::elems_per_thread>(), thread, input, instantiableOperationInstances...);
            }
        }
        // We pass TFI as a template parameter because sometimes we need to disable the TF
        template <typename TFI, typename ReadIOp>
        FK_DEVICE_FUSE auto read(const Point& thread, const ReadIOp& readDF) {
            if constexpr (TFI::ENABLED) {
                return ReadIOp::Operation::exec<TFI::elems_per_thread>(thread, readDF);
            } else {
                return ReadIOp::Operation::exec(thread, readDF);
            }
        }

        template <typename TFI, typename ReadIOp, typename... IOps>
        FK_DEVICE_FUSE 
        void execute_instantiable_operations_helper(const Point& thread, const ReadIOp& readDF,
                                                    const IOps&... iOps) {
            using ReadOperation = typename ReadIOp::Operation;
            using WriteOperation = typename LastType_t<IOps...>::Operation;

            const auto writeDF = ppLast(iOps...);

            if constexpr (TFI::ENABLED) {
                const auto tempI = read<TFI, ReadIOp>(thread, readDF);
                if constexpr (sizeof...(iOps) > 1) {
                    const auto tempO = operate_thread_fusion<TFI>(thread, tempI, iOps...);
                    WriteOperation::exec<TFI::elems_per_thread>(thread, tempO, writeDF);
                } else {
                    WriteOperation::exec<TFI::elems_per_thread>(thread, tempI, writeDF);
                }
            } else {
                const auto tempI = read<TFI, ReadIOp>(thread, readDF);
                if constexpr (sizeof...(iOps) > 1) {
                    const auto tempO = operate(thread, tempI, iOps...);
                    WriteOperation::exec(thread, tempO, writeDF);
                } else {
                    WriteOperation::exec(thread, tempI, writeDF);
                }
            }
        }

        template <typename TFI, typename... IOps>
        FK_DEVICE_FUSE
        void execute_instantiable_operations(const Point& thread,
                                             const IOps&... iOps){
            execute_instantiable_operations_helper<TFI>(thread, iOps...);
        }

        public:
        template <typename FirstIOp>
        FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const Details& details,
                                                           const FirstIOp& iOp) {
            if constexpr (Details::TFI::ENABLED) {
                return details.activeThreads;
            } else {
                return FirstIOp::Operation::getActiveThreads(iOp);
            }
        }

        template <typename... IOps>
        FK_DEVICE_FUSE void exec(const Details& details, const IOps&... iOps) {
            const cg::thread_block g = cg::this_thread_block();

            const int x = (g.dim_threads().x * g.group_index().x) + g.thread_index().x;
            const int y = (g.dim_threads().y * g.group_index().y) + g.thread_index().y;
            const int z = g.group_index().z; // So far we only consider the option of using the z dimension to specify n (x*y) thread planes
            const Point thread{ x, y, z };

            using TFI = typename Details::TFI;

            const ActiveThreads activeThreads = getActiveThreads(details, get<0>(iOps...));

            if (x < activeThreads.x && y < activeThreads.y) {
                if constexpr (!TFI::ENABLED) {
                    execute_instantiable_operations<TFI>(thread, iOps...);
                } else {
                    if constexpr (THREAD_DIVISIBLE) {
                        execute_instantiable_operations<TFI>(thread, iOps...);
                    } else {
                        const bool iamlastActiveThread = x == activeThreads.x - 1;
                        if (!iamlastActiveThread) {
                            execute_instantiable_operations<TFI>(thread, iOps...);
                        } else if (iamlastActiveThread) {
                            const int initialX = x * TFI::elems_per_thread;
                            using ReadOp = typename FirstType_t<IOps...>::Operation;
                            const int finalX = ReadOp::num_elems_x(thread, get<0>(iOps...));
                            int currentX = initialX;
                            while (currentX < finalX) {
                                const Point currentThread{ currentX , thread.y, thread.z };
                                using ReadIT = typename FirstType_t<IOps...>::Operation::ReadDataType;
                                using WriteOT = typename LastType_t<IOps...>::Operation::WriteDataType;
                                using DisabledTFI = ThreadFusionInfo<ReadIT, WriteOT, false>;
                                execute_instantiable_operations<DisabledTFI>(currentThread, iOps...);
                                currentX++;
                            }
                        }
                    }
                }
            }
        }
    };

    template <>
    struct TransformDPP<void, true> {
        template <bool THREAD_FUSION, typename FirstIOp, typename... IOps>
        FK_HOST_FUSE auto build_details(const FirstIOp& firstIOp, const IOps&... iOps) {
            using Details = TransformDPPDetails<THREAD_FUSION, FirstIOp, IOps...>;
            using TFI = typename Details::TFI;

            if constexpr (TFI::ENABLED) {
                const ActiveThreads initAT = firstIOp.getActiveThreads();
                const ActiveThreads gridActiveThreads(static_cast<uint>(ceil(initAT.x / static_cast<float>(TFI::elems_per_thread))),
                                                      initAT.y, initAT.z);
                const bool threadDivisible = isThreadDivisible<TFI::ENABLED>(TFI::elems_per_thread, firstIOp, iOps...);
                const Details details{ gridActiveThreads, threadDivisible };

                return details;
            } else {
                return Details{};
            }
        }
    };

    template <typename SequenceSelector>
    struct DivergentBatchTransformDPP {
    private:

        template <typename... IOps>
        FK_DEVICE_FUSE void launchTransformDPP(const IOps&... iOps) {
            using Details = TransformDPPDetails<false, IOps...>;
            TransformDPP<Details, true>::exec(Details{}, iOps...);
        }

        template <int OpSequenceNumber, typename... IOps, typename... IOpSequenceTypes>
        FK_DEVICE_FUSE void divergent_operate(const uint& z, const InstantiableOperationSequence<IOps...>& iOpSequence,
                                              const IOpSequenceTypes&... iOpSequences) {
            if (OpSequenceNumber == SequenceSelector::at(z)) {
                apply(launchTransformDPP<IOps...>, iOpSequence.instantiableOperations);
            } else if constexpr (sizeof...(iOpSequences) > 0) {
                divergent_operate<OpSequenceNumber + 1>(z, iOpSequences...);
            }
        }
    public:
        template <typename... IOpSequenceTypes>
        FK_DEVICE_FUSE void exec(const IOpSequenceTypes&... iOpSequences) {
            const cg::thread_block g = cg::this_thread_block();
            const uint z = g.group_index().z;
            divergent_operate<1>(z, iOpSequences...);
        }
    };

    template <typename SequenceSelector, typename... IOpSequences>
    __global__ void launchDivergentBatchTransformDPP_Kernel(const __grid_constant__ IOpSequences... iOpSequences) {
        DivergentBatchTransformDPP<SequenceSelector>::exec(iOpSequences...);
    }

    template <bool THREAD_DIVISIBLE, bool THREAD_FUSION, typename... IOps>
    __global__ void launchTransformDPP_Kernel(const __grid_constant__ TransformDPPDetails_<void,THREAD_FUSION, IOps...> tDPPDetails,
                                              const __grid_constant__ IOps... operations) {
        TransformDPP<TransformDPPDetails_<void, THREAD_FUSION, IOps...>, THREAD_DIVISIBLE>::exec(tDPPDetails, operations...);
    }
} // namespace fk

#endif
