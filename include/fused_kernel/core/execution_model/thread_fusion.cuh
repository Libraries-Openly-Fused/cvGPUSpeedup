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

#include <fused_kernel/core/utils/type_lists.cuh>
#include <fused_kernel/core/utils/cuda_vector_utils.cuh>
#include <fused_kernel/core/utils/template_operations.cuh>
#include <fused_kernel/core/data/vector_types.cuh>
#include <cuda_runtime.h>

namespace fk {

    /* Possible combinations:
    Size, Channels, Types
    1,    1         char, uchar                     8,  8  char8, uchar8
    2,    1         short, ushort                   8,  4  short4, ushort4
    4,    1         int, uint, float                8,  2  int2, uint2, float2
    8,    1         longlong, ulonglong, double     8,  1
    2,    2         char2, uchar2                   8,  8  char8, uchar8
    4,    2         short2, ushort2                 8,  4  short4, ushort4
    8,    2         int2, uint2, float2             8,  2
    16,   2         longlong2, ulonglong2, double2  16, 2
    3,    3         char3, uchar3                   12, 12 char12, uchar12
    6,    3         short3, ushort3                 24, 12 short12, ushort12
    12,   3         int3, uint3, float3             12, 3
    24,   3         longlong3, ulonglong3, double3  24, 3
    4,    4         char4, uchar4                   8,  8  char8, uchar8
    8,    4         short4, ushort4                 8,  4
    16,   4         int4, uint4, float4             16, 4
    32,   4         longlong4, ulonglong4, double4  32, 4

    Times bigger can be: 1, 2, 4, 8
    */

    using TFSourceTypes = typename TypeList<StandardTypes, VTwo, VThree, VFour>::type;
    using TFBiggerTypes = TypeList<uchar8,  char8,  ushort4,  short4,  uint2, int2, ulong,  long,  ulonglong,  longlong,  float2, double,
                                   uchar8,  char8,  ushort4,  short4,  uint2, int2, ulong2, long2, ulonglong2, longlong2, float2, double2,
                                   uchar12, char12, ushort12, short12, uint3, int3, ulong3, long3, ulonglong3, longlong3, float3, double3,
                                   uchar8,  char8,  ushort4,  short4,  uint4, int4, ulong4, long4, ulonglong4, longlong4, float4, double4>;
    template <typename SourceType>
    using TFBiggerType_t = EquivalentType_t<SourceType, TFSourceTypes, TFBiggerTypes>;

    template <typename OriginalType_t, bool ENABLED>
    struct ThreadFusionInfo {
        static constexpr bool ENABLED{ ENABLED };
        using OriginalType = OriginalType_t;
        using BiggerType = std::conditional_t<ENABLED, TFBiggerType_t<OriginalType>, OriginalType>;
        enum { times_bigger = cn<BiggerType> / cn<OriginalType> };
        template <int IDX>
        FK_HOST_DEVICE_FUSE OriginalType get(const BiggerType& data) {
            static_assert(IDX < times_bigger, "Index out of range for this ThreadFusionInfo");
            if constexpr (validCUDAVec<OriginalType>) {
                if constexpr (cn<OriginalType> == 2) {
                    if constexpr (IDX == 0) {
                        return make_<OriginalType>(data.x, data.y);
                    } else if constexpr (IDX == 1) {
                        return make_<OriginalType>(data.z, data.w);
                    } else if constexpr (IDX == 2) {
                        return make_<OriginalType>(data.i, data.j);
                    } else if constexpr (IDX == 3) {
                        return make_<OriginalType>(data.k, data.l);
                    }
                } else if constexpr (cn<OriginalType> == 3) {
                    if constexpr (IDX == 0) {
                        return make_<OriginalType>(data.x, data.y, data.z);
                    } else if constexpr (IDX == 1) {
                        return make_<OriginalType>(data.x1, data.y1, data.z1);
                    } else if constexpr (IDX == 2) {
                            return make_<OriginalType>(data.x2, data.y2, data.z2);
                    } else if constexpr (IDX == 3) {
                        return make_<OriginalType>(data.x3, data.y3, data.z3);
                    }
                } else if constexpr (cn<OriginalType> == 4) {
                    if constexpr (IDX == 0) {
                        return make_<OriginalType>(data.x, data.y, data.z, data.w);
                    } else if constexpr (IDX == 1) {
                        return make_<OriginalType>(data.i, data.j, data.k, data.l);
                    }
                }
            } else {
                if constexpr (IDX == 0) {
                    return data.x;
                } else if constexpr (IDX == 1) {
                    return data.y;
                } else if constexpr (IDX == 2) {
                    return data.z;
                } else if constexpr (IDX == 3) {
                    return data.w;
                } else if constexpr (IDX == 4) {
                    return data.i;
                } else if constexpr (IDX == 5) {
                    return data.j;
                } else if constexpr (IDX == 6) {
                    return data.k;
                } else if constexpr (IDX == 7) {
                    return data.l;
                }
            }
        }
        template <typename... OriginalTypes>
        FK_HOST_DEVICE_FUSE BiggerType make(const OriginalTypes&... data) {
            static_assert(and_v<std::is_same_v<OriginalType, OriginalTypes>...>, "Not all types are the same when making the ThreadFusion BiggerType value");
            if constexpr (cn<OriginalType> > 1) {
                return make_impl(data...);
            } else {
                return make_<BiggerType>(data...);
            }
        }

        private:
        FK_HOST_DEVICE_FUSE BiggerType make_impl(const OriginalType& data0,
                                                 const OriginalType& data1) {
            if constexpr (cn<OriginalType> == 2) {
                return make_<BiggerType>(data0.x, data0.y, data1.x, data1.y);
            } else if constexpr (cn<OriginalType> == 4) {
                return make_<BiggerType>(data0.x, data0.y, data0.z, data0.w,
                                         data1.x, data1.y, data1.z, data1.w);
            }
        }
        FK_HOST_DEVICE_FUSE BiggerType make_impl(const OriginalType& data0,
                                                 const OriginalType& data1,
                                                 const OriginalType& data2,
                                                 const OriginalType& data3) {
            if constexpr (cn<OriginalType> == 2) {
                return make_<BiggerType>(data0.x, data0.y, data1.x, data1.y,
                                         data2.x, data2.y, data3.x, data3.y);
            } else if constexpr (cn<OriginalType> == 3) {
                return make_<BiggerType>(data0.x, data0.y, data0.z,
                                         data1.x, data1.y, data1.z,
                                         data2.x, data2.y, data2.z,
                                         data3.x, data3.y, data3.z);
            }
        }
};

    template <typename... ThreadFusionInfos>
    constexpr bool allTFEnabled = (ThreadFusionInfos::ENABLED && ...);
} // namespace fk
