// A complex number.
#ifndef __MYCOMPLEX_H__
#define __MYCOMPLEX_H__

#include <gpu_macros.hpp>

template <typename T>
class Complex {
    public:
    T real;
    T imag;
    
    #ifdef __GPU__
    __host__ __device__
    #endif
    Complex operator+(const Complex<T>& other) const {
        return {real + other.real, imag + other.imag};
    }

    #ifdef __GPU__
    __host__ __device__
    #endif
    Complex& operator+=(const Complex<T>& other) {
        this->real += other.real;
        this->imag += other.imag;
        return *this;
    }
    #ifdef __GPU__
    __host__ __device__
    #endif
    Complex operator-(const Complex<T>& other) const {
        return {real - other.real, imag - other.imag};
    }

    template <typename T2>
    #ifdef __GPU__
    __host__ __device__
    #endif
    Complex operator/(const T2& other) const {
        return {real / other, imag / other};
    }

    Complex operator*(const Complex& other) const {
        Complex res { real * other.real - imag * other.imag, real * other.imag + imag * other.real};
        return res;
    }

    Complex conj() const {
        return {real, -imag};
    }

    inline friend std::istream& operator>>(std::istream& is, Complex<T>& m){
        is >> m.real >> m.imag;
        return is;
    }
    friend std::ostream& operator<<(std::ostream& os, const Complex<T>& p){
        os << "(" << p.real << ", " << p.imag << ")";
        return os; 
    }

    #ifdef __GPU__
    __host__ __device__
    #endif
    double magnitude() const {
        return std::sqrt(real * real + imag * imag);
    }
};

#endif
