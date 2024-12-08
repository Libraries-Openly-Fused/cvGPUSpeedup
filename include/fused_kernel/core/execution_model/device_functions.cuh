/* Copyright 2023-2024 Oscar Amoros Huguet

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

#include <vector_types.h>
#include <fused_kernel/core/execution_model/fused_operation.cuh>
#include <fused_kernel/core/utils/parameter_pack_utils.cuh>

namespace fk { // namespace FusedKernel

#define IS \
    template <typename IT> \
    static constexpr bool is{ std::is_same_v<IT, InstanceType> };

#define ASSERT(instance_type) \
    static_assert(std::is_same_v<typename Operation::InstanceType, instance_type>, "Operation is not " #instance_type );

#define DEVICE_FUNCTION_DETAILS(instance_type) \
    using Operation = Operation_t; \
    using InstanceType = instance_type;

#define DEVICE_FUNCTION_DETAILS_IS(instance_type) \
    using Operation = Operation_t; \
    using InstanceType = instance_type; \
    IS

#define IS_ASSERT(instance_type) \
    IS \
    ASSERT(instance_type)

#define DEVICE_FUNCTION_DETAILS_IS_ASSERT(instance_type) \
    DEVICE_FUNCTION_DETAILS(instance_type) \
    IS_ASSERT(instance_type)

    template <typename Operation_t>
    struct ReadDeviceFunction final : public OperationData<Operation_t> {
        DEVICE_FUNCTION_DETAILS_IS_ASSERT(ReadType)
        static constexpr bool isSource{false};
    };

    /**
    * @brief SourceReadDeviceFunction: represents a DeviceFunction that reads data from global memory and returns it in registers.
    * It uses the thread indexes, and an additional parameter which should contain the pointer or pointers from where to read.
    * Expects Operation_t to have an static __device__ function member with the following parameters:
    * OutputType exec(const Point& thread, const ParamsType& params).
    * It can only be the first DeviceFunction in a sequence of DeviceFunctions.
    */
    template <typename Operation_t>
    struct SourceReadDeviceFunction final : public OperationData<Operation_t> {
        DEVICE_FUNCTION_DETAILS_IS_ASSERT(ReadType)
        static constexpr bool isSource{ true };
        dim3 activeThreads;
    };

    template <typename Operation_t>
    struct ReadBackDeviceFunction final : public OperationData<Operation_t> {
        DEVICE_FUNCTION_DETAILS_IS_ASSERT(ReadBackType)
        static constexpr bool isSource{ false };
    };

    /**
    * @brief SourceReadBackDeviceFunction: represents a DeviceFunction that reads data from global memory and returns it in registers.
    * Additionally, it gets a DeviceFunction that it will use at some point of it's Operation implementation.
    * Usually, it will be another Read or ReadBack DeviceFunction.
    * Expects Operation_t to have an static __device__ function member with the following parameters:
    * OutputType exec(const Point& thread, const ParamsType& params, const BackFunction& back_function)
    * It can only be the first DeviceFunction in a sequence of DeviceFunctions.
    */
    template <typename Operation_t>
    struct SourceReadBackDeviceFunction final : public OperationData<Operation_t> {
        DEVICE_FUNCTION_DETAILS_IS_ASSERT(ReadBackType)
        static constexpr bool isSource{ true };
        dim3 activeThreads;
    };

    /**
    * @brief BinaryDeviceFunction_: implementation alias of BinaryDeviceFunction
    */
    template <typename Enabler, typename FirstOp, typename... Operations>
    struct BinaryDeviceFunction_ {};

    template <typename FirstOp, typename... Operations>
    struct BinaryDeviceFunction_<std::enable_if_t<(sizeof...(Operations) > 0)>, FirstOp, Operations...> final
        : OperationData<FusedOperation<FirstOp, Operations...>> {
        using Operation = FusedOperation<FirstOp, Operations...>;
        using InstanceType = BinaryType;
        IS_ASSERT(BinaryType)

        constexpr BinaryDeviceFunction_() {}

        template <typename... ParamTypes>
        constexpr BinaryDeviceFunction_(const ParamTypes&... provided_params)
            : OperationData<FusedOperation<FirstOp, Operations...>>(make_operation_tuple<FirstOp, Operations...>(provided_params...)) {}
    };

    template <typename FirstOp, typename... Operations>
    struct BinaryDeviceFunction_<std::enable_if_t<(sizeof...(Operations) == 0)>, FirstOp, Operations...> final
    : public OperationData<FirstOp> {
        using Operation = FirstOp;
        using InstanceType = BinaryType;
        IS_ASSERT(BinaryType)
    };

    /**
    * @brief BinaryDeviceFunction: represents a DeviceFunction that takes the result of the previous DeviceFunction as input
    * (which will reside on GPU registers) and an additional parameter that contains data not generated during the execution
    * of the current kernel.
    * It generates an output and returns it in register memory.
    * It can be composed of a single Operation or of a chain of Operations, in which case it wraps them into an
    * FusedOperation.
    * Expects Operation_t to have an static __device__ function member with the following parameters:
    * OutputType exec(const InputType& input, const ParamsType& params)
    */
    template <typename... Operations>
    using BinaryDeviceFunction = BinaryDeviceFunction_<void, Operations...>;

    /**
    * @brief TernaryDeviceFunction: represents a DeviceFunction that takes the result of the previous DeviceFunction as input
    * (which will reside on GPU registers) plus two additional parameters.
    * Second parameter (params): represents the same as in a BinaryFunction, data thas was not generated during the execution
    * of this kernel.
    * Third parameter (back_function): it's a DeviceFunction that will be used at some point in the implementation of the
    * Operation. It can be any kind of DeviceFunction.
    * Expects Operation_t to have an static __device__ function member with the following parameters:
    * OutputType exec(const InputType& input, const ParamsType& params, const BackFunction& back_function)
    */
    template <typename Operation_t>
    struct TernaryDeviceFunction final : public OperationData<Operation_t> {
        DEVICE_FUNCTION_DETAILS_IS_ASSERT(TernaryType)
    };

    /**
    * @brief UnaryDeviceFunction: represents a DeviceFunction that takes the result of the previous DeviceFunction as input
    * (which will reside on GPU registers).
    * It allows to execute the Operation (or chain of Unary Operations) on the input, and returns the result as output
    * in register memory.
    * Expects Operation_t to have an static __device__ function member with the following parameters:
    * OutputType exec(const InputType& input)
    */
    template <typename Enabler, typename FirstOperation, typename... Operations>
    struct UnaryDeviceFunction_;

    template <typename FirstOperation, typename... Operations>
    struct UnaryDeviceFunction_<std::enable_if_t<(sizeof...(Operations) > 0)>, FirstOperation, Operations...> {
        using Operation = FusedOperation<FirstOperation, Operations...>;
        using InstanceType = UnaryType;
        IS_ASSERT(UnaryType)
    };

    template <typename FirstOperation, typename... Operations>
    struct UnaryDeviceFunction_<std::enable_if_t<(sizeof...(Operations) == 0)>, FirstOperation, Operations...> {
        using Operation = FirstOperation;
        using InstanceType = UnaryType;
        IS_ASSERT(UnaryType)
    };

    template <typename... Operations>
    using UnaryDeviceFunction = UnaryDeviceFunction_<void, Operations...>;

    /**
    * @brief MidWriteDeviceFunction: represents a DeviceFunction that takes the result of the previous DeviceFunction as input
    * (which will reside on GPU registers) and writes it into device memory. The way that the data is written, is definded
    * by the implementation of Operation_t.
    * It returns the input data without modification, so that another DeviceFunction can be executed after it, using the same data.
    */
    template <typename Operation_t>
    struct MidWriteDeviceFunction final : public OperationData<Operation_t> {
        DEVICE_FUNCTION_DETAILS_IS(MidWriteType)
        static_assert(std::is_same_v<typename Operation::InstanceType, WriteType> ||
                      std::is_same_v<typename Operation::InstanceType, MidWriteType>,
                      "Operation is not WriteType or MidWriteType");
    };

    /**
    * @brief WriteDeviceFunction: represents a DeviceFunction that takes the result of the previous DeviceFunction as input
    * (which will reside on GPU registers) and writes it into device memory. The way that the data is written, is definded
    * by the implementation of Operation_t.
    * It can only be the last DeviceFunction in a sequence of DeviceFunctions.
    */
    template <typename Operation_t>
    struct WriteDeviceFunction final : public OperationData<Operation_t> {
        DEVICE_FUNCTION_DETAILS_IS_ASSERT(WriteType)
    };

#undef DEVICE_FUNCTION_DETAILS

    template <typename Operation>
    using SourceRead = SourceReadDeviceFunction<Operation>;
    template <typename Operation>
    using SourceReadBack = SourceReadBackDeviceFunction<Operation>;
    template <typename Operation>
    using Read = ReadDeviceFunction<Operation>;
    template <typename Operation>
    using ReadBack = ReadBackDeviceFunction<Operation>;
    template <typename... Operations>
    using Unary = UnaryDeviceFunction<Operations...>;
    template <typename... Operations>
    using Binary = BinaryDeviceFunction<Operations...>;
    template <typename Operation>
    using Ternary = TernaryDeviceFunction<Operation>;
    template <typename Operation>
    using MidWrite = MidWriteDeviceFunction<Operation>;
    template <typename Operation>
    using Write = WriteDeviceFunction<Operation>;

    template <typename DeviceFunction>
    FK_HOST_CNST auto make_source(const DeviceFunction& readDF, const dim3& activeThreads) {
        using Op = typename DeviceFunction::Operation;
        if constexpr (DeviceFunction::template is<ReadBackType>) {
            return SourceReadBack<Op>{readDF.params, readDF.back_function, activeThreads};
        } else {
            return SourceRead<Op>{readDF.params, activeThreads};
        }
    }

    template <typename Enabler, typename... Operations>
    struct DeviceFunctionType {};

    // Single Operation cases
    template <typename FirstType, typename... Operations>
    struct DeviceFunctionType<std::enable_if_t<sizeof...(Operations) == 0 &&
                                               isReadType<FirstType>>, FirstType, Operations...> {
        using Operation = FirstType;
        using type = Read<Operation>;
    };

    template <typename FirstType, typename... Operations>
    struct DeviceFunctionType<std::enable_if_t<sizeof...(Operations) == 0 &&
                                               isReadBackType<FirstType>>, FirstType, Operations...> {
        using Operation = FirstType;
        using type = ReadBack<Operation>;
    };

    template <typename FirstType, typename... Operations>
    struct DeviceFunctionType<std::enable_if_t<sizeof...(Operations) == 0 &&
                                               isTernaryType<FirstType>>, FirstType, Operations...> {
        using Operation = FirstType;
        using type = Ternary<Operation>;
    };

    template <typename FirstType, typename... Operations>
    struct DeviceFunctionType<std::enable_if_t<sizeof...(Operations) == 0 &&
                                               isWriteType<FirstType>>, FirstType, Operations...> {
        using Operation = FirstType;
        using type = Write<Operation>;
    };

    // Multiple Operation cases
    template <typename FirstType, typename... Operations>
    struct DeviceFunctionType<std::enable_if_t<(sizeof...(Operations) > 0) &&
                                               isAnyReadType<FirstType>>, FirstType, Operations...> {
        using type = Read<FusedOperation<FirstType, Operations...>>;
    };

    template <typename FirstType, typename... Operations>
    struct DeviceFunctionType<std::enable_if_t<(sizeof...(Operations) > 0) &&
                                               allUnaryTypes<FirstType, Operations...>>, FirstType, Operations...> {
        using type = Unary<FirstType, Operations...>;
    };

    template <typename FirstType, typename... Operations>
    struct DeviceFunctionType<std::enable_if_t<(sizeof...(Operations) > 0) &&
                                               !allUnaryTypes<FirstType, Operations...> &&
                                               isComputeType<FirstType>>, FirstType, Operations...> {
        using type = Binary<FirstType, Operations...>;
    };

    template <typename FirstType, typename... Operations>
    struct DeviceFunctionType<std::enable_if_t<(sizeof...(Operations) > 0) &&
                                               isWriteType<FirstType>>, FirstType, Operations...> {
        using type = MidWrite<FusedOperation<FirstType, Operations...>>;
    };

    template <typename... Operations>
    using DF_t = typename DeviceFunctionType<void, Operations...>::type;

    /** @brief fuseDF: function that creates either a Read or a Binary DeviceFunction, composed of an
    * OpertationTupleOperation (OTO), where the operations are the ones found in the DeviceFunctions in the
    * deviceFunctions parameter pack.
    * This is a convenience function to simplify the implementation of ReadBack and Ternary DeviceFunctions
    * and Operations.
    */
    template <typename... DeviceFunctions>
    FK_HOST_CNST auto fuseDF(const DeviceFunctions&... deviceFunctions) {
        using FirstDF = FirstType_t<DeviceFunctions...>;
        if constexpr (isAnyReadType<FirstDF> && FirstDF::isSource) {
            return make_source(DF_t<FusedOperation<typename DeviceFunctions::Operation...>>
                               { devicefunctions_to_operationtuple(deviceFunctions...) },
                               ppFirst(deviceFunctions...).activeThreads);
        } else {
            return DF_t<FusedOperation<typename DeviceFunctions::Operation...>>
                    { devicefunctions_to_operationtuple(deviceFunctions...) };
        }
    }

    template <typename Operation, size_t NPtr, size_t... Index>
    constexpr inline std::array<DF_t<Operation>, NPtr> paramsArrayToDFArray_helper(
        const std::array<typename Operation::ParamsType, NPtr>& paramsArr,
        const std::index_sequence<Index...>&) 
    {
        return { (DF_t<Operation>{paramsArr[Index]})... };
    }

    template <typename Operation, size_t NPtr>
    constexpr inline std::array<DF_t<Operation>, NPtr> paramsArrayToDFArray(
        const std::array<typename Operation::ParamsType, NPtr>& paramsArr) 
    {
        return paramsArrayToDFArray_helper<Operation>(paramsArr, std::make_index_sequence<NPtr>{});
    }

    template <typename Operation, size_t NPtr, size_t... Index>
    constexpr inline std::array<DF_t<Operation>, NPtr> paramsArrayToDFArray_helper(
        const std::array<typename Operation::ParamsType, NPtr>& paramsArr,
        const std::array<typename Operation::BackFunction, NPtr>& backFunctions,
        const std::index_sequence<Index...>&) {
        return { (DF_t<Operation>{ OperationData<Operation>{paramsArr[Index], backFunctions[Index]} })... };
    }

    template <typename Operation, size_t NPtr>
    constexpr inline std::array<DF_t<Operation>, NPtr> paramsArrayToDFArray(
        const std::array<typename Operation::ParamsType, NPtr>& paramsArr,
        const std::array<typename Operation::BackFunction, NPtr>& backFunctions) {
        return paramsArrayToDFArray_helper<Operation>(paramsArr, backFunctions, std::make_index_sequence<NPtr>{});
    }
} // namespace fk
