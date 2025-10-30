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

#include <iostream>
#include <opencv2/core.hpp>

#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/utils/utils.h>
#include <fused_kernel/core/utils/vector_utils.h>

template <int Depth> std::string depthToString() { return ""; }

#define DEPTH_TO_STRING(cv_depth, string_t)                                    \
  template <> std::string depthToString<cv_depth>() { return string_t; }

DEPTH_TO_STRING(CV_8U, "CV_8U")
DEPTH_TO_STRING(CV_8S, "CV_8S")
DEPTH_TO_STRING(CV_16U, "CV_16U")
DEPTH_TO_STRING(CV_16S, "CV_16S")
DEPTH_TO_STRING(CV_32S, "CV_32S")
DEPTH_TO_STRING(CV_32F, "CV_32F")
DEPTH_TO_STRING(CV_64F, "CV_64F")
#undef DEPTH_TO_STRING

template <int Channels> std::string channelsToString() { return ""; }

#define CHANNELS_TO_STRING(cv_channels, string_t)                              \
  template <> std::string channelsToString<cv_channels>() { return string_t; }

CHANNELS_TO_STRING(1, "C1")
CHANNELS_TO_STRING(2, "C2")
CHANNELS_TO_STRING(3, "C3")
CHANNELS_TO_STRING(4, "C4")
#undef CHANNELS_TO_STRING

template <int T> std::string cvTypeToString() {
  return depthToString<CV_MAT_DEPTH(T)>() + channelsToString<CV_MAT_CN(T)>();
}

using SourceType = fk::TypeList<uchar, char, ushort, short, uint, int, ulong, long, ulonglong, longlong, float, double>;
using PrintableType = fk::TypeList<uint, int, uint, int, uint, int, ulong, long, ulonglong, longlong, float, double>;

template <typename T> auto getPrintableValue(const T &value) {
  return static_cast<fk::EquivalentType_t<T, SourceType, PrintableType>>(value);
}

template <typename T> void printV(T value) {
  if constexpr (fk::Channels<T>() >= 1) {
    if constexpr (fk::validCUDAVec<T>) {
      const auto printable_value = getPrintableValue(value.x);
      std::cout << "(" << printable_value;
    } else {
      const auto printable_value = getPrintableValue(value);
      std::cout << "(" << printable_value;
    }
  }
  if constexpr (fk::Channels<T>() >= 2) {
    std::cout << ", " << getPrintableValue(value.y);
  }
  if constexpr (fk::Channels<T>() >= 3) {
    std::cout << ", " << getPrintableValue(value.z);
  }
  if constexpr (fk::Channels<T>() == 4) {
    std::cout << ", " << getPrintableValue(value.w);
  }
  std::cout << ")";
}

template <typename T>
void print2D(const std::string &message, const fk::Ptr2D<T> &d_input, const cudaStream_t &compute_stream) {
  fk::Ptr2D<T> h_input(d_input.dims().width, d_input.dims().height, 0, fk::MemType::HostPinned);
  gpuErrchk(cudaMemcpy2DAsync(h_input.ptr().data, h_input.dims().pitch, d_input.ptr().data, d_input.dims().pitch,
                              d_input.dims().width * sizeof(T), d_input.dims().height, cudaMemcpyDeviceToHost,
                              compute_stream));
  gpuErrchk(cudaStreamSynchronize(compute_stream));

  std::cout << std::endl;

  std::cout << message << std::endl;
  for (int y = 0; y < d_input.dims().height; y++) {
    for (int x = 0; x < d_input.dims().width; x++) {
      const T value = *fk::PtrAccessor<fk::ND::_2D>::cr_point(fk::Point(x, y), h_input.ptr());
      printV(value);
      std::cout << ",";
    }
    std::cout << std::endl;
  }
}

namespace fk {
template <typename T> void printTensor(const fk::Tensor<T> &tensor) {
  std::stringstream ss;

  const auto dims = tensor.dims();
  const size_t plane_pixels = dims.width * dims.height;

  for (int z = 0; z < dims.planes; z++) {
    for (int y = 0; y < dims.height; y++) {
      for (int cp = 0; cp < dims.color_planes; cp++) {
        const T *plane = fk::PtrAccessor<fk::ND::_3D>::cr_point(fk::Point(0, 0, z), tensor.ptr()) + (plane_pixels * cp);

        for (int x = 0; x < dims.width; x++) {
          ss << plane[x + (y * dims.width)] << " ";
        }
        ss << "| ";
      }
      ss << std::endl;
    }
    ss << "------------------" << std::endl;
  }
  std::cout << ss.str() << std::endl;
}

template <typename T> void printTensor(const fk::TensorT<T> &tensor) {
  std::stringstream ss;

  const auto dims = tensor.dims();
  const size_t plane_pixels = dims.width * dims.height;

  for (int cp = 0; cp < dims.color_planes; cp++) {
    for (int y = 0; y < dims.height; y++) {
      for (int z = 0; z < dims.planes; z++) {
        const T *plane =
            fk::PtrAccessor<fk::ND::T3D>::cr_point(fk::Point(0, 0, z), tensor.ptr()) + (plane_pixels * dims.planes * cp);
        for (int x = 0; x < dims.width; x++) {
          ss << plane[x + (y * dims.width)] << " ";
        }
        ss << "| ";
      }
      ss << std::endl;
    }
    ss << "------------------" << std::endl;
  }
  std::cout << ss.str() << std::endl;
}

template <typename T> void printTensorImagePerRow(const fk::Tensor<T> &tensor) {
  const auto dims = tensor.dims();
  const size_t elements_per_image = dims.width * dims.height * dims.color_planes;
  for (int i = 0; i < tensor.getNumElements(); i++) {
    if (i > 0 && i % elements_per_image == 0) {
      std::cout << std::endl;
    }
    std::cout << tensor.ptr().data[i];
  }
  std::cout << std::endl;
}

template <typename T> void printTensorImagePerRow(const fk::TensorT<T> &tensor) {
  const auto dims = tensor.dims();
  const size_t elements_per_image = dims.width * dims.height * dims.planes;
  for (int i = 0; i < tensor.getNumElements(); i++) {
    if (i > 0 && i % elements_per_image == 0) {
      std::cout << std::endl;
    }
    std::cout << tensor.ptr().data[i];
  }
  std::cout << std::endl;
}

template <typename TensorType> bool compareTensors(const TensorType &tensor1, const TensorType &tensor2) {
  bool correct = tensor1.getNumElements() == tensor2.getNumElements();
  if (!correct) {
    std::cout << "Tensors don't have the same number of elements!!" << std::endl;
  }
  for (int i = 0; i < tensor1.getNumElements(); i++) {
    correct &= tensor1.ptr().data[i] == tensor2.ptr().data[i];
  }
  return correct;
}
} // namespace fk