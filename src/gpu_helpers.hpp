#include <gpu_macros.hpp>
#include <mycomplex.hpp>

/**
 * @brief Compute the conjugate cross multiply (ccm) between std::complex numbers `a` and `b`
 * and add the result to the `res` accumulator.
 */
template <typename T1, typename T2>
__host__ __device__ __forceinline__ void ccm(
    const Complex<T1>& a,
    const Complex<T1>& b,
    Complex<T2>& acc
) {
    T2 ar = static_cast<T2>(a.real);
    T2 ai = static_cast<T2>(a.imag);
    T2 br = static_cast<T2>(b.real);
    T2 bi = static_cast<T2>(b.imag);

    acc.real += ar * br + ai * bi;
    acc.imag += ai * br - ar * bi;
}

/**
 * @brief Simple helper to accumulate complex values accross warps using shfl_down.
 */
__device__ __forceinline__ Complex<float> warp_sum(Complex<float> acc) {
    for (unsigned int i = warpSize/2; i > 0; i >>= 1) {
        acc.real += __gpu_shfl_down(acc.real, i);
        acc.imag += __gpu_shfl_down(acc.imag, i);
    }
    return acc;
}

/**
 * @brief Reinterpret (type pun) helper.
 *
 * This function uses `__builtin_memcpy` rather than a `reinterpret_cast<>`, a
 * standard pattern for type-punning that preserves strict aliasing rules. The
 * compiler recognises this pattern and optimises away the memcpy.
 *
 * More info: https://en.cppreference.com/w/cpp/language/reinterpret_cast.html#Type_aliasing
 */
template <typename Tout, typename Tin>
__device__ __forceinline__ Tout type_alias(Tin in) {
    Tout out;
    __builtin_memcpy(&out, &in, sizeof(out));
    return out;
}

/**
 * @brief Dot products four 8-bit integers in pairs (each packed into a 32-bit int)
 * and accumulates the result.
 */
__device__ __forceinline__ int dp4a(int a, int b, int acc) {
    #if defined(__HIPCC__)
    return __builtin_amdgcn_sdot4(a, b, acc, false);
    #elif defined(__NVCC__)
    return __dp4a(a, b, acc);
    #else
    char4 a4 = type_alias<char4>(a);
    char4 b4 = type_alias<char4>(b);
    acc += a4.x * b4.x;
    acc += a4.y * b4.y;
    acc += a4.z * b4.z;
    acc += a4.w * b4.w;
    return acc;
    #endif
}

/**
 * @brief Load two contiguous complex samples, type punning through a single
 *  32-bit int.
 */
__device__ __forceinline__ int load_32bits(const Complex<int8_t>* p) {
    int x;
    __builtin_memcpy(&x, p, sizeof(x));
    return x;
}

/**
 * Perform a complex conjugate multiplication on two pairs of complex samples
 *  that have been packed into two 32-bit integers (each sample is 2x 8-bits).
 */
__forceinline__ __device__ void ccm_dp4a(int A, int B, Complex<int>& acc) {
    // unpack 32-bit int into 4 x 8-bit
    char4 b = type_alias<char4>(B);

    // constructing the conjugate terms
    char4 bi;
    bi.x = (signed char)(-b.y);
    bi.y = b.x;
    bi.z = (signed char)(-b.w);
    bi.w = b.z;

    // re-packing into 32-bits
    int Bi = type_alias<int>(bi);

    acc.real = dp4a(A, B, acc.real);
    acc.imag = dp4a(A, Bi, acc.imag);
}