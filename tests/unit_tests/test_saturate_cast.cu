/* Copyright 2025 Grup Mediapro S.L.U.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include <tests/main.h>
#include <tests/testsCommon.cuh>

#include <cvGPUSpeedup.cuh>
#include <fused_kernel/core/utils/type_to_string.h>
#include "opencv2/imgproc.hpp"
#include <opencv2/core/cuda/vec_math.hpp>

#include <iomanip>

bool debugT{ true };

template <typename TypeTo, typename TypeFrom>
bool cpuSaturateCast(const TypeFrom& source) {
    const bool res = fk::SaturateCast<TypeFrom, TypeTo>::exec(source) == cv::saturate_cast<TypeTo>(source);
    if (!res && debugT) {
        std::cout << "Mismatch for " << std::to_string(source) << " with " << fk::typeToString<TypeFrom>()
            << " to " << fk::typeToString<TypeTo>() << std::endl;
    }
        
    return res;
}

namespace fk {
    template <typename I, typename O>
    struct OCVSaturateCast {
    private:
        using SelfType = OCVSaturateCast<I, O>;
    public:
        FK_STATIC_STRUCT(OCVSaturateCast, SelfType)
        using Parent = UnaryOperation<I, O, OCVSaturateCast<I, O>>;
        DECLARE_UNARY_PARENT
        FK_DEVICE_FUSE OutputType exec(const InputType& input) {
            return cv::cuda::device::saturate_cast<OutputType>(input);
        }
    };
}

template <typename TypeTo, typename TypeFrom>
bool gpuSaturateCast(const TypeFrom& source, fk::Stream_<fk::ParArch::GPU_NVIDIA> stream) {
    static_assert(fk::cn<TypeTo> == fk::cn<TypeFrom>, "Types must have same number of channels");

    fk::Ptr1D<TypeFrom> d_input(1, 0, fk::MemType::DeviceAndPinned);
    fk::Ptr1D<TypeTo> d_output_ocv(1, 0, fk::MemType::DeviceAndPinned);
    fk::Ptr1D<TypeTo> d_output_cvGS(1, 0, fk::MemType::DeviceAndPinned);

    d_input.at(0) = source;
    d_input.upload(stream);

    auto readOp = fk::PerThreadRead<fk::ND::_1D, TypeFrom>::build(d_input.ptr());
    auto ocvSaturateCastOp = fk::OCVSaturateCast<TypeFrom, TypeTo>::build();
    auto cvGSSaturateCastOp = fk::SaturateCast<TypeFrom, TypeTo>::build();
    auto ocvWriteOp = fk::PerThreadWrite<fk::ND::_1D, TypeTo>::build(d_output_ocv.ptr());
    auto cvGSWriteOp = fk::PerThreadWrite<fk::ND::_1D, TypeTo>::build(d_output_cvGS.ptr());

    fk::executeOperations<fk::TransformDPP<>>(stream, readOp, ocvSaturateCastOp, ocvWriteOp);
    fk::executeOperations<fk::TransformDPP<>>(stream, readOp, cvGSSaturateCastOp, cvGSWriteOp);
    d_output_ocv.download(stream);
    d_output_cvGS.download(stream);
    
    stream.sync();

    const auto resv = d_output_ocv.at(0) == d_output_cvGS.at(0);

    const bool res = resv;

    if (!res && debugT) {
        if constexpr (fk::cn<TypeFrom> == 1) {

        } else if constexpr (fk::cn<TypeFrom> == 2) {

        } else if constexpr (fk::cn<TypeFrom> == 3) {
            std::cout << std::fixed << std::setprecision(9);
            std::cout << "Mismatch for [" << source.x << ", " <<
                source.y << ", " <<
                source.z << "] with " << fk::typeToString<TypeFrom>()
                << " to " << fk::typeToString<TypeTo>() << std::endl;
            std::cout << "OCV result: [" << static_cast<int>(d_output_ocv.at(0).x) << ", " <<
                static_cast<int>(d_output_ocv.at(0).y) << ", " <<
                static_cast<int>(d_output_ocv.at(0).z) << "]" << std::endl;
            std::cout << "cvGS result: [" << static_cast<int>(d_output_cvGS.at(0).x) << ", " <<
                static_cast<int>(d_output_cvGS.at(0).y) << ", " <<
                static_cast<int>(d_output_cvGS.at(0).z) << "]" << std::endl;
        } else {

        }
    }

    return res;
}

bool test_cpuSaturateCast() {
    bool ok{ true };

    //Rounding test
    ok &= cpuSaturateCast<uint8_t>(2.3f);
    ok &= cpuSaturateCast<uint8_t>(2.49f);
    ok &= cpuSaturateCast<uint8_t>(2.5f);
    ok &= cpuSaturateCast<uint8_t>(3.5f);
    ok &= cpuSaturateCast<uint8_t>(2.51f);
    ok &= cpuSaturateCast<uint8_t>(2.7f);
    ok &= cpuSaturateCast<int16_t>(2.3f);
    ok &= cpuSaturateCast<int16_t>(2.49f);
    ok &= cpuSaturateCast<int16_t>(2.5f);
    ok &= cpuSaturateCast<int16_t>(3.5f);
    ok &= cpuSaturateCast<int16_t>(2.51f);
    ok &= cpuSaturateCast<int16_t>(2.7f);
    ok &= cpuSaturateCast<int16_t>(-2.3f);
    ok &= cpuSaturateCast<int16_t>(-2.49f);
    ok &= cpuSaturateCast<int16_t>(-2.5f);
    ok &= cpuSaturateCast<int16_t>(-3.5f);
    ok &= cpuSaturateCast<int16_t>(-2.51f);
    ok &= cpuSaturateCast<int16_t>(-2.7f);
    ok &= cpuSaturateCast<int>(2.3f);
    ok &= cpuSaturateCast<int>(2.49f);
    ok &= cpuSaturateCast<int>(2.5f);
    ok &= cpuSaturateCast<int>(3.5f);
    ok &= cpuSaturateCast<int>(2.51f);
    ok &= cpuSaturateCast<int>(2.7f);
    ok &= cpuSaturateCast<int>(-2.3f);
    ok &= cpuSaturateCast<int>(-2.49f);
    ok &= cpuSaturateCast<int>(-2.5f);
    ok &= cpuSaturateCast<int>(-2.51f);
    ok &= cpuSaturateCast<int>(-2.7f);
    ok &= cpuSaturateCast<uint>(2.3f);
    ok &= cpuSaturateCast<uint>(2.49f);
    ok &= cpuSaturateCast<uint>(2.5f);
    ok &= cpuSaturateCast<uint>(3.5f);
    ok &= cpuSaturateCast<uint>(2.51f);
    ok &= cpuSaturateCast<uint>(2.7f);
    ok &= cpuSaturateCast<uchar>(1.999999881f);
    ok &= cpuSaturateCast<uchar>(36.999996185f);
    ok &= cpuSaturateCast<uchar>(127.999992371f);

    // GPU tests
    fk::Stream stream;
    ok &= gpuSaturateCast<uchar3>(float3{ 1.999999881f, 36.999996185f, 127.999992371f }, stream);
    ok &= gpuSaturateCast<uchar3>(float3{ 1.49f, 36.5f, 127.51f }, stream);

    constexpr float3 f3{ 1.999999881f, 36.999996185f, 127.999992371f };
    constexpr float3 result = cxp::nearbyint::f(f3);
    static_assert(result == float3{ 2.f, 37.f, 128.f }, "nearbyint failed");
   
    // Test int to uchar
    for (int i = -10; i <= 300; ++i) {
        ok &= cpuSaturateCast<uint8_t>(i);
    }
    // Test float to uchar
    for (float f = -10.0f; f <= 300.0f; f += 1.0f) {
        ok &= cpuSaturateCast<uint8_t>(f);
    }
    // Test double to uchar
    for (double d = -10.0; d <= 300.0; d += 1.0) {
        ok &= cpuSaturateCast<uint8_t>(d);
    }
    // Test int to short
    for (int i = -40000; i <= 40000; i += 1000) {
        ok &= cpuSaturateCast<int16_t>(i);
    }
    // Test float to short
    for (float f = -40000.0f; f <= 40000.0f; f += 1000.0f) {
        ok &= cpuSaturateCast<int16_t>(f);
    }
    // Test double to short
    for (double d = -40000.0; d <= 40000.0; d += 1000.0) {
        ok &= cpuSaturateCast<int16_t>(d);
    }
    return ok;
}

int launch() {
    bool ok = test_cpuSaturateCast();
    int result{ 0 };
    if (ok) {
        std::cout << "SaturateCast tests passed successfully!" << std::endl;
    } else {
        std::cout << "SaturateCast tests failed!" << std::endl;
        result = -1;
    }

    return result;
}