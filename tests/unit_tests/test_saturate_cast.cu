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