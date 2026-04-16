/* Copyright 2025 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef REALBATCH_H
#define REALBATCH_H

#include <cstddef>

constexpr size_t REAL_BATCH = 1;
#ifndef CUDART_MAJOR_VERSION
#error CUDART_MAJOR_VERSION Undefined!
#elif (CUDART_MAJOR_VERSION == 11)
constexpr size_t NUM_EXPERIMENTS = 5;
constexpr size_t FIRST_VALUE = 2;
constexpr size_t INCREMENT = 50;
#elif (CUDART_MAJOR_VERSION == 12)
constexpr size_t NUM_EXPERIMENTS = CPP_NUM_EXPERIMENTS;
constexpr size_t FIRST_VALUE = 2;
constexpr size_t INCREMENT = 100;
#endif

#endif // REALBATCH_H