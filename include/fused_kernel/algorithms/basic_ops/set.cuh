/* Copyright 2024 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_SET
#define FK_SET

#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/execution_model/operation_types.cuh>
#include <fused_kernel/core/execution_model/default_builders_def.h>

namespace fk {
    template <typename T>
    struct ReadSet {
        using InstanceType = ReadType;
        using OutputType = T;
        using ParamsType = T;
        using ReadDataType = T;
        static constexpr bool THREAD_FUSION{ false };
        FK_HOST_DEVICE_FUSE OutputType exec(const Point& thread, const ParamsType& params) {
            return params;
        }
        using InstantiableType = Read<ReadSet<T>>;
        DEFAULT_READ_BUILD
    };
} // namespace fk

#include <fused_kernel/core/execution_model/default_builders_undef.h>

#endif
