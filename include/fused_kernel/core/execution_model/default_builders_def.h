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

#ifndef BUILDERS
#define BUILDERS

#define DEFAULT_READ_BATCH_BUILD \
private: \
template <size_t Idx, typename Array> \
FK_HOST_FUSE auto get_element_at_index(const Array& paramArray) -> decltype(paramArray[Idx]) { \
    return paramArray[Idx]; \
} \
template <size_t Idx, typename... Arrays> \
FK_HOST_FUSE auto call_build_at_index(const Arrays&... arrays) { \
    return build(get_element_at_index<Idx>(arrays)...); \
} \
template <size_t... Idx, typename... Arrays> \
FK_HOST_FUSE auto build_helper_generic(const std::index_sequence<Idx...>&, \
                                       const Arrays&... arrays) { \
    using OutputArrayType = decltype(call_build_at_index<0>(std::declval<Arrays>()...)); \
    return std::array<OutputArrayType, sizeof...(Idx)>{ call_build_at_index<Idx>(arrays...)... }; \
} \
public: \
template <size_t BATCH_N, typename FirstType, typename... ArrayTypes> \
FK_HOST_FUSE auto \
build_batch(const std::array<FirstType, BATCH_N>& firstInstance, const ArrayTypes&... arrays) { \
    static_assert(allArraysSameSize_v<BATCH_N, ArrayTypes...>, \
                  "Not all arrays have the same size as BATCH"); \
    return build_helper_generic(std::make_index_sequence<BATCH_N>(), firstInstance, arrays...); \
} \
template <size_t BATCH_N, typename FirstType, typename... ArrayTypes> \
FK_HOST_FUSE auto build(const std::array<FirstType, BATCH_N>& firstInstance, \
                        const ArrayTypes&... arrays) { \
    const auto arrayOfIOps = build_batch(firstInstance, arrays...); \
    return BatchRead<BATCH_N>::build(arrayOfIOps); \
} \
template <size_t BATCH_N, typename DefaultValueType, typename FirstType, typename... ArrayTypes> \
FK_HOST_FUSE auto build(const int& usedPlanes, const DefaultValueType& defaultValue, \
                        const std::array<FirstType, BATCH_N>& firstInstance, \
                        const ArrayTypes&... arrays) { \
    const auto arrayOfIOps = build_batch(firstInstance, arrays...); \
    return BatchRead<BATCH_N, CONDITIONAL_WITH_DEFAULT>::build(arrayOfIOps, usedPlanes, defaultValue); \
}

#define DEFAULT_WRITE_BATCH_BUILD \
private: \
template <size_t Idx, typename Array> \
FK_HOST_FUSE auto get_element_at_index(const Array& paramArray) -> decltype(paramArray[Idx]) { \
    return paramArray[Idx]; \
} \
template <size_t Idx, typename... Arrays> \
FK_HOST_FUSE auto call_build_at_index(const Arrays&... arrays) { \
    return build(get_element_at_index<Idx>(arrays)...); \
} \
template <size_t... Idx, typename... Arrays> \
FK_HOST_FUSE auto build_helper_generic(const std::index_sequence<Idx...>&, \
                                       const Arrays&... arrays) { \
    using OutputArrayType = decltype(call_build_at_index<0>(std::declval<Arrays>()...)); \
    return std::array<OutputArrayType, sizeof...(Idx)>{ call_build_at_index<Idx>(arrays...)... }; \
} \
public: \
template <size_t BATCH_N, typename FirstType, typename... ArrayTypes> \
FK_HOST_FUSE auto \
build_batch(const std::array<FirstType, BATCH_N>& firstInstance, const ArrayTypes&... arrays) { \
    static_assert(allArraysSameSize_v<BATCH_N, ArrayTypes...>, \
                  "Not all arrays have the same size as BATCH"); \
    return build_helper_generic(std::make_index_sequence<BATCH_N>(), firstInstance, arrays...); \
} \
template <size_t BATCH_N, typename FirstType, typename... ArrayTypes> \
FK_HOST_FUSE auto build(const std::array<FirstType, BATCH_N>& firstInstance, \
                        const ArrayTypes&... arrays) { \
    const auto arrayOfIOps = build_batch(firstInstance, arrays...); \
    return BatchWrite<BATCH_N>::build(arrayOfIOps); \
}

#define DEFAULT_BUILD \
FK_HOST_DEVICE_FUSE auto build(const OperationDataType& opData) { \
    return InstantiableType{ opData }; \
}

#define DEFAULT_BUILD_PARAMS \
FK_HOST_FUSE auto build(const ParamsType& params) { \
    return InstantiableType{ {params} }; \
}

#define DEFAULT_BUILD_PARAMS_BACKIOP \
FK_HOST_FUSE auto build(const ParamsType& params, const BackFunction& backIOp) { \
    return InstantiableType{ {params, backIOp} }; \
}

#define DEFAULT_UNARY_BUILD \
FK_HOST_DEVICE_FUSE auto build() { \
    return InstantiableType{}; \
}

#endif
