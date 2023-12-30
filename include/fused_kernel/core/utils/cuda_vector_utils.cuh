/* Copyright 2023 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#pragma once

#include <fused_kernel/core/utils/cuda_utils.cuh>
#include <fused_kernel/core/utils/type_lists.cuh>
#include <fused_kernel/core/data/vector_types.cuh>
#include <fused_kernel/core/utils/template_operations.cuh>

namespace fk {

    template <typename BaseType, int Channels>
    struct VectorType {};

#define VECTOR_TYPE(BaseType) \
    template <> \
    struct VectorType<BaseType, 1> { using type = BaseType; }; \
    template <> \
    struct VectorType<BaseType, 2> { using type = BaseType ## 2; }; \
    template <> \
    struct VectorType<BaseType, 3> { using type = BaseType ## 3; }; \
    template <> \
    struct VectorType<BaseType, 4> { using type = BaseType ## 4; };

    VECTOR_TYPE(uchar)
    VECTOR_TYPE(char)
    VECTOR_TYPE(short)
    VECTOR_TYPE(ushort)
    VECTOR_TYPE(int)
    VECTOR_TYPE(uint)
    VECTOR_TYPE(long)
    VECTOR_TYPE(ulong)
    VECTOR_TYPE(longlong)
    VECTOR_TYPE(ulonglong)
    VECTOR_TYPE(float)
    VECTOR_TYPE(double)
#undef VECTOR_TYPE

    template <typename BaseType, int Channels>
    using VectorType_t = typename VectorType<BaseType, Channels>::type;

    template <uint CHANNELS>
    using VectorTypeList = TypeList<VectorType_t<uchar, CHANNELS>, VectorType_t<char, CHANNELS>,
                                    VectorType_t<ushort, CHANNELS>, VectorType_t<short,CHANNELS>,
                                    VectorType_t<uint, CHANNELS>, VectorType_t<int, CHANNELS>,
                                    VectorType_t<ulong, CHANNELS>, VectorType_t<long, CHANNELS>,
                                    VectorType_t<ulonglong, CHANNELS>, VectorType_t<longlong, CHANNELS>,
                                    VectorType_t<float, CHANNELS>, VectorType_t<double, CHANNELS>>;

    using StandardTypes =
        TypeList<uchar, char, ushort, short, uint, int, ulong, long, ulonglong, longlong, float, double>;
    using VOne = TypeList<uchar1, char1, ushort1, short1, uint1, int1, ulong1, long1, ulonglong1, longlong1, float1, double1>;
    using VTwo = VectorTypeList<2>;
    using VThree = VectorTypeList<3>;
    using VFour = VectorTypeList<4>;
    using VAll   = typename TypeList<VOne, VTwo, VThree, VFour>::type;

    template <typename T>
    constexpr bool validCUDAVec = one_of<T, VAll>::value;

    template <typename T>
    __host__ __device__ __forceinline__ constexpr int Channels() {
        if constexpr (one_of_v<T, VOne> || !validCUDAVec<T>) {
            return 1;
        } else if constexpr (one_of_v<T, VTwo>) { 
            return 2;
        } else if constexpr (one_of_v<T, VThree>) { 
            return 3;
        } else if constexpr (one_of_v<T, VFour>) {
            return 4;
        }
    }

    template <typename T>
    constexpr int cn = Channels<T>();

    template <int idx, typename T>
    __host__ __device__ __forceinline__ constexpr auto VectorAt(const T& vector) {
        static_assert(validCUDAVec<T>, "Non valid CUDA vetor type: VectorAt<invalid_type>()");
        if constexpr (idx == 0) { 
            return vector.x; 
        } else if constexpr (idx == 1) { 
            static_assert(cn<T> >= 2, "Vector type smaller than 2 elements has no member y"); 
            return vector.y;
        } else if constexpr (idx == 2) { 
            static_assert(cn<T> >= 3, "Vector type smaller than 3 elements has no member z");
            return vector.z;
        } else if constexpr (idx == 3) {
            static_assert(cn<T> == 4, "Vector type smaller than 4 elements has no member w");
            return vector.w;
        }
    }

    template <int... idx>
    struct VReorder {
        template <typename T>
        FK_HOST_DEVICE_FUSE T exec(const T& vector) {
            static_assert(validCUDAVec<T>, "Non valid CUDA vetor type: VReorder<...>::exec<invalid_type>(invalid_type vector)");
            static_assert(sizeof...(idx) == cn<T>, "Wrong number of indexes for the cuda vetor type in VReorder.");
            return {VectorAt<idx>(vector)...};
        }
    };

    template <typename V>
    struct VectorTraits {};

#define VECTOR_TRAITS(BaseType) \
    template <> \
    struct VectorTraits<BaseType> { using base = BaseType; enum {bytes=sizeof(base)}; }; \
    template <> \
    struct VectorTraits<BaseType ## 1> { using base = BaseType; enum {bytes=sizeof(base)}; }; \
    template <> \
    struct VectorTraits<BaseType ## 2> { using base = BaseType; enum {bytes=sizeof(base)*2}; }; \
    template <> \
    struct VectorTraits<BaseType ## 3> { using base = BaseType; enum {bytes=sizeof(base)*3}; }; \
    template <> \
    struct VectorTraits<BaseType ## 4> { using base = BaseType; enum {bytes=sizeof(base)*4}; };

    VECTOR_TRAITS(uchar)
    VECTOR_TRAITS(char)
    VECTOR_TRAITS(short)
    VECTOR_TRAITS(ushort)
    VECTOR_TRAITS(int)
    VECTOR_TRAITS(uint)
    VECTOR_TRAITS(long)
    VECTOR_TRAITS(ulong)
    VECTOR_TRAITS(longlong)
    VECTOR_TRAITS(ulonglong)
    VECTOR_TRAITS(float)
    VECTOR_TRAITS(double)
#undef VECTOR_TRAITS

    template <typename T>
    using VBase = typename VectorTraits<T>::base;

    // Automagically making any CUDA vector type from a template type
    // It will not compile if you try to do bad things. The number of elements
    // need to conform to T, and the type of the elements will always be casted.
    struct make {
        template <typename T, typename... Numbers>
        FK_HOST_DEVICE_FUSE T type(const Numbers&... pack) {
            static_assert(validCUDAVec<T>, "Non valid CUDA vetor type: make::type<invalid_type>()");
            return {static_cast<decltype(T::x)>(pack)...};
        }
    };

    template <typename T, typename... Numbers>
    FK_HOST_DEVICE_CNST T make_(const Numbers&... pack) {
        if constexpr (std::is_aggregate_v<T>) {
            return make::type<T>(pack...);
        } else {
            static_assert(sizeof...(pack) == 1, "Something wrong in make_");
            return first(pack...);
        }
    }
    
    template <typename T, typename Enabler=void>
    struct UnaryVectorSet;
    
    template <typename T>
    struct UnaryVectorSet<T, typename std::enable_if_t<!validCUDAVec<T>>>{
        // This case exists to make things easier when we don't know if the type
        // is going to be a vector type or a normal type
        FK_HOST_DEVICE_FUSE T exec(const T& val) {
            return val;
        }
    };

    template <typename T>
    struct UnaryVectorSet<T, typename std::enable_if_t<validCUDAVec<T>>> {
        FK_HOST_DEVICE_FUSE T exec(const typename VectorTraits<T>::base& val) {
            if constexpr (cn<T> == 1) {
                return {val};
            } else if constexpr (cn<T> == 2) {
                return {val, val};
            } else if constexpr (cn<T> == 3) {
                return {val, val, val};
            } else {
                return {val, val, val, val};
            }
        }
    };

    template <typename T>
    __device__ __forceinline__ __host__ constexpr T make_set(const typename VectorTraits<T>::base& val) {
        return UnaryVectorSet<T>::exec(val);
    }

    template <typename T>
    __device__ __forceinline__ __host__ constexpr T make_set(const T& val) {
        return UnaryVectorSet<T>::exec(val);
    }

    template <typename T>
    struct to_printable {
        FK_HOST_FUSE int exec(T val) {
            if constexpr (sizeof(T) == 1) {
                return static_cast<int>(val);
            } else if constexpr (sizeof(T) > 1) {
                return val;
            }
        }
    };

    template <typename T>
    struct print_vector {
        FK_HOST_FUSE std::ostream& exec(std::ostream& outs, T val) {
            if constexpr (!validCUDAVec<T>) {
                outs << val;
                return outs;
            } else if constexpr (cn<T> == 1) {
                outs << "{" << to_printable<decltype(T::x)>::exec(val.x) << "}";
                return outs;
            } else if constexpr (cn<T> == 2) {
                outs << "{" << to_printable<decltype(T::x)>::exec(val.x) <<
                       ", " << to_printable<decltype(T::y)>::exec(val.y) << "}";
                return outs;
            } else if constexpr (cn<T> == 3) {
                outs << "{" << to_printable<decltype(T::x)>::exec(val.x) <<
                       ", " << to_printable<decltype(T::y)>::exec(val.y) <<
                       ", " << to_printable<decltype(T::z)>::exec(val.z) << "}";
                return outs;
            } else {
                 outs << "{" << to_printable<decltype(T::x)>::exec(val.x) <<
                        ", " << to_printable<decltype(T::y)>::exec(val.y) <<
                        ", " << to_printable<decltype(T::z)>::exec(val.z) <<
                        ", " << to_printable<decltype(T::w)>::exec(val.w) << "}";
                return outs;
            }
        }
    };

    template <typename T> 
    __host__ inline constexpr typename std::enable_if_t<validCUDAVec<T>, std::ostream&> operator<<(std::ostream& outs, const T& val) {
        return print_vector<T>::exec(outs, val);
    }

    namespace vec_math_detail
    {
        template <int cn, typename VecD> struct SatCastHelper;
        template <typename VecD> struct SatCastHelper<1, VecD>
        {
            template <typename VecS> static __device__ __forceinline__ constexpr VecD cast(const VecS& v)
            {
                using D = typename VectorTraits<VecD>::base;
                return make::type<VecD>(saturate_cast<D>(v.x));
            }
        };
        template <typename VecD> struct SatCastHelper<2, VecD>
        {
            template <typename VecS> static __device__ __forceinline__ constexpr VecD cast(const VecS& v)
            {
                using D = typename VectorTraits<VecD>::base;
                return make::type<VecD>(saturate_cast<D>(v.x), saturate_cast<D>(v.y));
            }
        };
        template <typename VecD> struct SatCastHelper<3, VecD>
        {
            template <typename VecS> static __device__ __forceinline__ constexpr VecD cast(const VecS& v)
            {
                using D = typename VectorTraits<VecD>::base;
                return make::type<VecD>(saturate_cast<D>(v.x), saturate_cast<D>(v.y), saturate_cast<D>(v.z));
            }
        };
        template <typename VecD> struct SatCastHelper<4, VecD>
        {
            template <typename VecS> static __device__ __forceinline__ constexpr VecD cast(const VecS& v)
            {
                using D = typename VectorTraits<VecD>::base;
                return make::type<VecD>(saturate_cast<D>(v.x), saturate_cast<D>(v.y), saturate_cast<D>(v.z), saturate_cast<D>(v.w));
            }
        };

        template <typename VecD, typename VecS> static __device__ __forceinline__ constexpr VecD saturate_cast_helper(const VecS& v)
        {
            return SatCastHelper<cn<VecD>, VecD>::cast(v);
        }
    }

    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const uchar1& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const char1& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const ushort1& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const short1& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const uint1& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const int1& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const float1& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const double1& v) { return vec_math_detail::saturate_cast_helper<T>(v); }

    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const uchar2& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const char2& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const ushort2& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const short2& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const uint2& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const int2& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const float2& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const double2& v) { return vec_math_detail::saturate_cast_helper<T>(v); }

    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const uchar3& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const char3& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const ushort3& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const short3& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const uint3& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const int3& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const float3& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const double3& v) { return vec_math_detail::saturate_cast_helper<T>(v); }

    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const uchar4& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const char4& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const ushort4& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const short4& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const uint4& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const int4& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const float4& v) { return vec_math_detail::saturate_cast_helper<T>(v); }
    template<typename T> static __device__ __forceinline__ constexpr T saturate_cast(const double4& v) { return vec_math_detail::saturate_cast_helper<T>(v); }

#define VEC_UNARY_OP(op, input_type, output_type) \
    __device__ __forceinline__ __host__ constexpr output_type ## 1 operator op(const input_type ## 1 & a) \
    { \
        return make::type<output_type ## 1>(op (a.x)); \
    } \
    __device__ __forceinline__ __host__ constexpr output_type ## 2 operator op(const input_type ## 2 & a) \
    { \
        return make::type<output_type ## 2>(op (a.x), op (a.y)); \
    } \
    __device__ __forceinline__ __host__ constexpr output_type ## 3 operator op(const input_type ## 3 & a) \
    { \
        return make::type<output_type ## 3>(op (a.x), op (a.y), op (a.z)); \
    } \
    __device__ __forceinline__ __host__ constexpr output_type ## 4 operator op(const input_type ## 4 & a) \
    { \
        return make::type<output_type ## 4>(op (a.x), op (a.y), op (a.z), op (a.w)); \
    }

    VEC_UNARY_OP(-, char, char)
    VEC_UNARY_OP(-, short, short)
    VEC_UNARY_OP(-, int, int)
    VEC_UNARY_OP(-, float, float)
    VEC_UNARY_OP(-, double, double)

    VEC_UNARY_OP(!, uchar, uchar)
    VEC_UNARY_OP(!, char, uchar)
    VEC_UNARY_OP(!, ushort, uchar)
    VEC_UNARY_OP(!, short, uchar)
    VEC_UNARY_OP(!, int, uchar)
    VEC_UNARY_OP(!, uint, uchar)
    VEC_UNARY_OP(!, float, uchar)
    VEC_UNARY_OP(!, double, uchar)

    VEC_UNARY_OP(~, uchar, uchar)
    VEC_UNARY_OP(~, char, char)
    VEC_UNARY_OP(~, ushort, ushort)
    VEC_UNARY_OP(~, short, short)
    VEC_UNARY_OP(~, int, int)
    VEC_UNARY_OP(~, uint, uint)

#undef VEC_UNARY_OP

        // binary operators (vec & vec)

#define VEC_BINARY_OP_DIFF_TYPES(op, input_type1, input_type2, output_type) \
    __device__ __forceinline__ __host__ constexpr output_type ## 1 operator op(const input_type1 ## 1 & a, const input_type2 ## 1 & b) \
    { \
        return make::type<output_type ## 1>(a.x op b.x); \
    } \
    __device__ __forceinline__ __host__ constexpr output_type ## 2 operator op(const input_type1 ## 2 & a, const input_type2 ## 2 & b) \
    { \
        return make::type<output_type ## 2>(a.x op b.x, a.y op b.y); \
    } \
    __device__ __forceinline__ __host__ constexpr output_type ## 3 operator op(const input_type1 ## 3 & a, const input_type2 ## 3 & b) \
    { \
        return make::type<output_type ## 3>(a.x op b.x, a.y op b.y, a.z op b.z); \
    } \
    __device__ __forceinline__ __host__ constexpr output_type ## 4 operator op(const input_type1 ## 4 & a, const input_type2 ## 4 & b) \
    { \
        return make::type<output_type ## 4>(a.x op b.x, a.y op b.y, a.z op b.z, a.w op b.w); \
    }

    VEC_BINARY_OP_DIFF_TYPES(+, uchar, float, float)

#undef VEC_BINARY_OP_DIFF_TYPES

#define VEC_BINARY_OP(op, input_type, output_type) \
    __device__ __forceinline__ __host__ constexpr output_type ## 1 operator op(const input_type ## 1 & a, const input_type ## 1 & b) \
    { \
        return make::type<output_type ## 1>(a.x op b.x); \
    } \
    __device__ __forceinline__ __host__ constexpr output_type ## 2 operator op(const input_type ## 2 & a, const input_type ## 2 & b) \
    { \
        return make::type<output_type ## 2>(a.x op b.x, a.y op b.y); \
    } \
    __device__ __forceinline__ __host__ constexpr output_type ## 3 operator op(const input_type ## 3 & a, const input_type ## 3 & b) \
    { \
        return make::type<output_type ## 3>(a.x op b.x, a.y op b.y, a.z op b.z); \
    } \
    __device__ __forceinline__ __host__ constexpr output_type ## 4 operator op(const input_type ## 4 & a, const input_type ## 4 & b) \
    { \
        return make::type<output_type ## 4>(a.x op b.x, a.y op b.y, a.z op b.z, a.w op b.w); \
    }

    VEC_BINARY_OP(+, uchar, int)
    VEC_BINARY_OP(+, char, int)
    VEC_BINARY_OP(+, ushort, int)
    VEC_BINARY_OP(+, short, int)
    VEC_BINARY_OP(+, int, int)
    VEC_BINARY_OP(+, uint, uint)
    VEC_BINARY_OP(+, float, float)
    VEC_BINARY_OP(+, double, double)

    VEC_BINARY_OP(-, uchar, int)
    VEC_BINARY_OP(-, char, int)
    VEC_BINARY_OP(-, ushort, int)
    VEC_BINARY_OP(-, short, int)
    VEC_BINARY_OP(-, int, int)
    VEC_BINARY_OP(-, uint, uint)
    VEC_BINARY_OP(-, float, float)
    VEC_BINARY_OP(-, double, double)

    VEC_BINARY_OP(*, uchar, int)
    VEC_BINARY_OP(*, char, int)
    VEC_BINARY_OP(*, ushort, int)
    VEC_BINARY_OP(*, short, int)
    VEC_BINARY_OP(*, int, int)
    VEC_BINARY_OP(*, uint, uint)
    VEC_BINARY_OP(*, float, float)
    VEC_BINARY_OP(*, double, double)

    VEC_BINARY_OP(/ , uchar, int)
    VEC_BINARY_OP(/ , char, int)
    VEC_BINARY_OP(/ , ushort, int)
    VEC_BINARY_OP(/ , short, int)
    VEC_BINARY_OP(/ , int, int)
    VEC_BINARY_OP(/ , uint, uint)
    VEC_BINARY_OP(/ , float, float)
    VEC_BINARY_OP(/ , double, double)

    VEC_BINARY_OP(== , uchar, uchar)
    VEC_BINARY_OP(== , char, uchar)
    VEC_BINARY_OP(== , ushort, uchar)
    VEC_BINARY_OP(== , short, uchar)
    VEC_BINARY_OP(== , int, uchar)
    VEC_BINARY_OP(== , uint, uchar)
    VEC_BINARY_OP(== , long, uchar)
    VEC_BINARY_OP(== , ulong, uchar)
    VEC_BINARY_OP(== , longlong, uchar)
    VEC_BINARY_OP(== , ulonglong, uchar)
    VEC_BINARY_OP(== , float, uchar)
    VEC_BINARY_OP(== , double, uchar)

    VEC_BINARY_OP(!= , uchar, uchar)
    VEC_BINARY_OP(!= , char, uchar)
    VEC_BINARY_OP(!= , ushort, uchar)
    VEC_BINARY_OP(!= , short, uchar)
    VEC_BINARY_OP(!= , int, uchar)
    VEC_BINARY_OP(!= , uint, uchar)
    VEC_BINARY_OP(!= , float, uchar)
    VEC_BINARY_OP(!= , double, uchar)

    VEC_BINARY_OP(> , uchar, uchar)
    VEC_BINARY_OP(> , char, uchar)
    VEC_BINARY_OP(> , ushort, uchar)
    VEC_BINARY_OP(> , short, uchar)
    VEC_BINARY_OP(> , int, uchar)
    VEC_BINARY_OP(> , uint, uchar)
    VEC_BINARY_OP(> , float, uchar)
    VEC_BINARY_OP(> , double, uchar)

    VEC_BINARY_OP(< , uchar, uchar)
    VEC_BINARY_OP(< , char, uchar)
    VEC_BINARY_OP(< , ushort, uchar)
    VEC_BINARY_OP(< , short, uchar)
    VEC_BINARY_OP(< , int, uchar)
    VEC_BINARY_OP(< , uint, uchar)
    VEC_BINARY_OP(< , float, uchar)
    VEC_BINARY_OP(< , double, uchar)

    VEC_BINARY_OP(>= , uchar, uchar)
    VEC_BINARY_OP(>= , char, uchar)
    VEC_BINARY_OP(>= , ushort, uchar)
    VEC_BINARY_OP(>= , short, uchar)
    VEC_BINARY_OP(>= , int, uchar)
    VEC_BINARY_OP(>= , uint, uchar)
    VEC_BINARY_OP(>= , float, uchar)
    VEC_BINARY_OP(>= , double, uchar)

    VEC_BINARY_OP(<= , uchar, uchar)
    VEC_BINARY_OP(<= , char, uchar)
    VEC_BINARY_OP(<= , ushort, uchar)
    VEC_BINARY_OP(<= , short, uchar)
    VEC_BINARY_OP(<= , int, uchar)
    VEC_BINARY_OP(<= , uint, uchar)
    VEC_BINARY_OP(<= , float, uchar)
    VEC_BINARY_OP(<= , double, uchar)

    VEC_BINARY_OP(&&, uchar, uchar)
    VEC_BINARY_OP(&&, char, uchar)
    VEC_BINARY_OP(&&, ushort, uchar)
    VEC_BINARY_OP(&&, short, uchar)
    VEC_BINARY_OP(&&, int, uchar)
    VEC_BINARY_OP(&&, uint, uchar)
    VEC_BINARY_OP(&&, float, uchar)
    VEC_BINARY_OP(&&, double, uchar)

    VEC_BINARY_OP(|| , uchar, uchar)
    VEC_BINARY_OP(|| , char, uchar)
    VEC_BINARY_OP(|| , ushort, uchar)
    VEC_BINARY_OP(|| , short, uchar)
    VEC_BINARY_OP(|| , int, uchar)
    VEC_BINARY_OP(|| , uint, uchar)
    VEC_BINARY_OP(|| , float, uchar)
    VEC_BINARY_OP(|| , double, uchar)

    VEC_BINARY_OP(&, uchar, uchar)
    VEC_BINARY_OP(&, char, char)
    VEC_BINARY_OP(&, ushort, ushort)
    VEC_BINARY_OP(&, short, short)
    VEC_BINARY_OP(&, int, int)
    VEC_BINARY_OP(&, uint, uint)

    VEC_BINARY_OP(| , uchar, uchar)
    VEC_BINARY_OP(| , char, char)
    VEC_BINARY_OP(| , ushort, ushort)
    VEC_BINARY_OP(| , short, short)
    VEC_BINARY_OP(| , int, int)
    VEC_BINARY_OP(| , uint, uint)

    VEC_BINARY_OP(^, uchar, uchar)
    VEC_BINARY_OP(^, char, char)
    VEC_BINARY_OP(^, ushort, ushort)
    VEC_BINARY_OP(^, short, short)
    VEC_BINARY_OP(^, int, int)
    VEC_BINARY_OP(^, uint, uint)

#undef VEC_BINARY_OP

        // binary operators (vec & scalar)

#define SCALAR_BINARY_OP(op, input_type, scalar_type, output_type) \
    __device__ __forceinline__ __host__ constexpr output_type ## 1 operator op(const input_type ## 1 & a, scalar_type s) \
    { \
        return make::type<output_type ## 1>(a.x op s); \
    } \
    __device__ __forceinline__ __host__ constexpr output_type ## 1 operator op(scalar_type s, const input_type ## 1 & b) \
    { \
        return make::type<output_type ## 1>(s op b.x); \
    } \
    __device__ __forceinline__ __host__ constexpr output_type ## 2 operator op(const input_type ## 2 & a, scalar_type s) \
    { \
        return make::type<output_type ## 2>(a.x op s, a.y op s); \
    } \
    __device__ __forceinline__ __host__ constexpr output_type ## 2 operator op(scalar_type s, const input_type ## 2 & b) \
    { \
        return make::type<output_type ## 2>(s op b.x, s op b.y); \
    } \
    __device__ __forceinline__ __host__ constexpr output_type ## 3 operator op(const input_type ## 3 & a, scalar_type s) \
    { \
        return make::type<output_type ## 3>(a.x op s, a.y op s, a.z op s); \
    } \
    __device__ __forceinline__ __host__ constexpr output_type ## 3 operator op(scalar_type s, const input_type ## 3 & b) \
    { \
        return make::type<output_type ## 3>(s op b.x, s op b.y, s op b.z); \
    } \
    __device__ __forceinline__ __host__ constexpr output_type ## 4 operator op(const input_type ## 4 & a, scalar_type s) \
    { \
        return make::type<output_type ## 4>(a.x op s, a.y op s, a.z op s, a.w op s); \
    } \
    __device__ __forceinline__ __host__ constexpr output_type ## 4 operator op(scalar_type s, const input_type ## 4 & b) \
    { \
        return make::type<output_type ## 4>(s op b.x, s op b.y, s op b.z, s op b.w); \
    }

    SCALAR_BINARY_OP(+, uchar, int, int)
    SCALAR_BINARY_OP(+, uchar, float, float)
    SCALAR_BINARY_OP(+, uchar, double, double)
    SCALAR_BINARY_OP(+, char, int, int)
    SCALAR_BINARY_OP(+, char, float, float)
    SCALAR_BINARY_OP(+, char, double, double)
    SCALAR_BINARY_OP(+, ushort, int, int)
    SCALAR_BINARY_OP(+, ushort, float, float)
    SCALAR_BINARY_OP(+, ushort, double, double)
    SCALAR_BINARY_OP(+, short, int, int)
    SCALAR_BINARY_OP(+, short, float, float)
    SCALAR_BINARY_OP(+, short, double, double)
    SCALAR_BINARY_OP(+, int, int, int)
    SCALAR_BINARY_OP(+, int, float, float)
    SCALAR_BINARY_OP(+, int, double, double)
    SCALAR_BINARY_OP(+, uint, uint, uint)
    SCALAR_BINARY_OP(+, uint, float, float)
    SCALAR_BINARY_OP(+, uint, double, double)
    SCALAR_BINARY_OP(+, float, float, float)
    SCALAR_BINARY_OP(+, float, double, double)
    SCALAR_BINARY_OP(+, double, double, double)

    SCALAR_BINARY_OP(-, uchar, int, int)
    SCALAR_BINARY_OP(-, uchar, float, float)
    SCALAR_BINARY_OP(-, uchar, double, double)
    SCALAR_BINARY_OP(-, char, int, int)
    SCALAR_BINARY_OP(-, char, float, float)
    SCALAR_BINARY_OP(-, char, double, double)
    SCALAR_BINARY_OP(-, ushort, int, int)
    SCALAR_BINARY_OP(-, ushort, float, float)
    SCALAR_BINARY_OP(-, ushort, double, double)
    SCALAR_BINARY_OP(-, short, int, int)
    SCALAR_BINARY_OP(-, short, float, float)
    SCALAR_BINARY_OP(-, short, double, double)
    SCALAR_BINARY_OP(-, int, int, int)
    SCALAR_BINARY_OP(-, int, float, float)
    SCALAR_BINARY_OP(-, int, double, double)
    SCALAR_BINARY_OP(-, uint, uint, uint)
    SCALAR_BINARY_OP(-, uint, float, float)
    SCALAR_BINARY_OP(-, uint, double, double)
    SCALAR_BINARY_OP(-, float, float, float)
    SCALAR_BINARY_OP(-, float, double, double)
    SCALAR_BINARY_OP(-, double, double, double)

    SCALAR_BINARY_OP(*, uchar, int, int)
    SCALAR_BINARY_OP(*, uchar, float, float)
    SCALAR_BINARY_OP(*, uchar, double, double)
    SCALAR_BINARY_OP(*, char, int, int)
    SCALAR_BINARY_OP(*, char, float, float)
    SCALAR_BINARY_OP(*, char, double, double)
    SCALAR_BINARY_OP(*, ushort, int, int)
    SCALAR_BINARY_OP(*, ushort, float, float)
    SCALAR_BINARY_OP(*, ushort, double, double)
    SCALAR_BINARY_OP(*, short, int, int)
    SCALAR_BINARY_OP(*, short, float, float)
    SCALAR_BINARY_OP(*, short, double, double)
    SCALAR_BINARY_OP(*, int, int, int)
    SCALAR_BINARY_OP(*, int, float, float)
    SCALAR_BINARY_OP(*, int, double, double)
    SCALAR_BINARY_OP(*, uint, uint, uint)
    SCALAR_BINARY_OP(*, uint, float, float)
    SCALAR_BINARY_OP(*, uint, double, double)
    SCALAR_BINARY_OP(*, float, float, float)
    SCALAR_BINARY_OP(*, float, double, double)
    SCALAR_BINARY_OP(*, double, double, double)

    SCALAR_BINARY_OP(/ , uchar, int, int)
    SCALAR_BINARY_OP(/ , uchar, float, float)
    SCALAR_BINARY_OP(/ , uchar, double, double)
    SCALAR_BINARY_OP(/ , char, int, int)
    SCALAR_BINARY_OP(/ , char, float, float)
    SCALAR_BINARY_OP(/ , char, double, double)
    SCALAR_BINARY_OP(/ , ushort, int, int)
    SCALAR_BINARY_OP(/ , ushort, float, float)
    SCALAR_BINARY_OP(/ , ushort, double, double)
    SCALAR_BINARY_OP(/ , short, int, int)
    SCALAR_BINARY_OP(/ , short, float, float)
    SCALAR_BINARY_OP(/ , short, double, double)
    SCALAR_BINARY_OP(/ , int, int, int)
    SCALAR_BINARY_OP(/ , int, float, float)
    SCALAR_BINARY_OP(/ , int, double, double)
    SCALAR_BINARY_OP(/ , uint, uint, uint)
    SCALAR_BINARY_OP(/ , uint, float, float)
    SCALAR_BINARY_OP(/ , uint, double, double)
    SCALAR_BINARY_OP(/ , float, float, float)
    SCALAR_BINARY_OP(/ , float, double, double)
    SCALAR_BINARY_OP(/ , double, double, double)

    SCALAR_BINARY_OP(== , uchar, uchar, uchar)
    SCALAR_BINARY_OP(== , char, char, uchar)
    SCALAR_BINARY_OP(== , ushort, ushort, uchar)
    SCALAR_BINARY_OP(== , short, short, uchar)
    SCALAR_BINARY_OP(== , int, int, uchar)
    SCALAR_BINARY_OP(== , uint, uint, uchar)
    SCALAR_BINARY_OP(== , float, float, uchar)
    SCALAR_BINARY_OP(== , double, double, uchar)

    SCALAR_BINARY_OP(!= , uchar, uchar, uchar)
    SCALAR_BINARY_OP(!= , char, char, uchar)
    SCALAR_BINARY_OP(!= , ushort, ushort, uchar)
    SCALAR_BINARY_OP(!= , short, short, uchar)
    SCALAR_BINARY_OP(!= , int, int, uchar)
    SCALAR_BINARY_OP(!= , uint, uint, uchar)
    SCALAR_BINARY_OP(!= , float, float, uchar)
    SCALAR_BINARY_OP(!= , double, double, uchar)

    SCALAR_BINARY_OP(> , uchar, uchar, uchar)
    SCALAR_BINARY_OP(> , char, char, uchar)
    SCALAR_BINARY_OP(> , ushort, ushort, uchar)
    SCALAR_BINARY_OP(> , short, short, uchar)
    SCALAR_BINARY_OP(> , int, int, uchar)
    SCALAR_BINARY_OP(> , uint, uint, uchar)
    SCALAR_BINARY_OP(> , float, float, uchar)
    SCALAR_BINARY_OP(> , double, double, uchar)

    SCALAR_BINARY_OP(< , uchar, uchar, uchar)
    SCALAR_BINARY_OP(< , char, char, uchar)
    SCALAR_BINARY_OP(< , ushort, ushort, uchar)
    SCALAR_BINARY_OP(< , short, short, uchar)
    SCALAR_BINARY_OP(< , int, int, uchar)
    SCALAR_BINARY_OP(< , uint, uint, uchar)
    SCALAR_BINARY_OP(< , float, float, uchar)
    SCALAR_BINARY_OP(< , double, double, uchar)

    SCALAR_BINARY_OP(>= , uchar, uchar, uchar)
    SCALAR_BINARY_OP(>= , char, char, uchar)
    SCALAR_BINARY_OP(>= , ushort, ushort, uchar)
    SCALAR_BINARY_OP(>= , short, short, uchar)
    SCALAR_BINARY_OP(>= , int, int, uchar)
    SCALAR_BINARY_OP(>= , uint, uint, uchar)
    SCALAR_BINARY_OP(>= , float, float, uchar)
    SCALAR_BINARY_OP(>= , double, double, uchar)

    SCALAR_BINARY_OP(<= , uchar, uchar, uchar)
    SCALAR_BINARY_OP(<= , char, char, uchar)
    SCALAR_BINARY_OP(<= , ushort, ushort, uchar)
    SCALAR_BINARY_OP(<= , short, short, uchar)
    SCALAR_BINARY_OP(<= , int, int, uchar)
    SCALAR_BINARY_OP(<= , uint, uint, uchar)
    SCALAR_BINARY_OP(<= , float, float, uchar)
    SCALAR_BINARY_OP(<= , double, double, uchar)

    SCALAR_BINARY_OP(&&, uchar, uchar, uchar)
    SCALAR_BINARY_OP(&&, char, char, uchar)
    SCALAR_BINARY_OP(&&, ushort, ushort, uchar)
    SCALAR_BINARY_OP(&&, short, short, uchar)
    SCALAR_BINARY_OP(&&, int, int, uchar)
    SCALAR_BINARY_OP(&&, uint, uint, uchar)
    SCALAR_BINARY_OP(&&, float, float, uchar)
    SCALAR_BINARY_OP(&&, double, double, uchar)

    SCALAR_BINARY_OP(|| , uchar, uchar, uchar)
    SCALAR_BINARY_OP(|| , char, char, uchar)
    SCALAR_BINARY_OP(|| , ushort, ushort, uchar)
    SCALAR_BINARY_OP(|| , short, short, uchar)
    SCALAR_BINARY_OP(|| , int, int, uchar)
    SCALAR_BINARY_OP(|| , uint, uint, uchar)
    SCALAR_BINARY_OP(|| , float, float, uchar)
    SCALAR_BINARY_OP(|| , double, double, uchar)

    SCALAR_BINARY_OP(&, uchar, uchar, uchar)
    SCALAR_BINARY_OP(&, char, char, char)
    SCALAR_BINARY_OP(&, ushort, ushort, ushort)
    SCALAR_BINARY_OP(&, short, short, short)
    SCALAR_BINARY_OP(&, int, int, int)
    SCALAR_BINARY_OP(&, uint, uint, uint)

    SCALAR_BINARY_OP(| , uchar, uchar, uchar)
    SCALAR_BINARY_OP(| , char, char, char)
    SCALAR_BINARY_OP(| , ushort, ushort, ushort)
    SCALAR_BINARY_OP(| , short, short, short)
    SCALAR_BINARY_OP(| , int, int, int)
    SCALAR_BINARY_OP(| , uint, uint, uint)

    SCALAR_BINARY_OP(^, uchar, uchar, uchar)
    SCALAR_BINARY_OP(^, char, char, char)
    SCALAR_BINARY_OP(^, ushort, ushort, ushort)
    SCALAR_BINARY_OP(^, short, short, short)
    SCALAR_BINARY_OP(^, int, int, int)
    SCALAR_BINARY_OP(^, uint, uint, uint)

#undef SCALAR_BINARY_OP

    template <uint ELEMS_PER_THREAD>
    struct SubVector {
        template <uint IDX, typename Vector>
        static __device__ __forceinline__ __host__ constexpr 
        auto get(const Vector& data) {
            static_assert(IDX < ELEMS_PER_THREAD, "SubVector index out of range");
            using OutputType = VectorType_t<VBase<Vector>, cn<Vector> / ELEMS_PER_THREAD>;
            getImpl<Vector, OutputType>(data,
                make_integer_sequence_from<uint, IDX * ELEMS_PER_THREAD, cn<OutputType>>());
        }
    private:
        template <typename Vector, typename OutputType, uint... IDX>
        static __device__ __forceinline__ __host__ constexpr
        OutputType getImpl(const Vector& data, std::integer_sequence<uint, IDX...>) {
            return make_<OutputType>(VectorAt<IDX, VBase<Vector>>(data)...);
        }
    };

}
