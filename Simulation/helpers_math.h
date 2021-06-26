#pragma once
#include <cuda_runtime.h>
#include <math.h>

// Negation
inline __host__ __device__ double3 operator-(double3& a)
{
    return make_double3(-a.x, -a.y, -a.z);
}

// Addition
inline __host__ __device__ double3 operator+(double3 a, double3 b)
{
    return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__ void operator+=(double3& a, double3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

// Subraction
inline __host__ __device__ double3 operator-(double3 a, double3 b)
{
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ void operator-=(double3& a, double3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}

// Multiplication
inline __host__ __device__ double3 operator*(double3 a, double3 b)
{
    return make_double3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline __host__ __device__ void operator*=(double3& a, double3 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}

inline __host__ __device__ double3 operator*(double3 a, float b)
{
    return make_double3(a.x * b, a.y * b, a.z * b);
}

inline __host__ __device__ double3 operator*(float b, double3 a)
{
    return make_double3(b * a.x, b * a.y, b * a.z);
}

inline __host__ __device__ void operator*=(double3& a, float b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}


// Division
inline __host__ __device__ double3 operator/(double3 a, double3 b)
{
    return make_double3(a.x / b.x, a.y / b.y, a.z / b.z);
}

inline __host__ __device__ void operator/=(double3& a, double3 b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
}

inline __host__ __device__ double3 operator/(double3 a, float b)
{
    return make_double3(a.x / b, a.y / b, a.z / b);
}

inline __host__ __device__ void operator/=(double3& a, float b)
{
    a.x /= b;
    a.y /= b;
    a.z /= b;
}

inline __host__ __device__ double3 operator/(float b, double3 a)
{
    return make_double3(b / a.x, b / a.y, b / a.z);
}

// Dot product
inline __host__ __device__ double dot(double3 a, double3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Absolute value
inline __host__ __device__ double3 fabs(double3 v)
{
    return make_double3(fabs(v.x), fabs(v.y), fabs(v.z));
}

// Norm
inline __host__ __device__ double norm(double3 v)
{
    return sqrt(dot(v, v));
}

// Norm squared for optimization
inline __host__ __device__ double norm_sq(double3 v)
{
    return dot(v, v);
}

// Round
inline __host__ __device__ double3 round(double3 v)
{
    return make_double3(round(v.x), round(v.y), round(v.z));
}

// Floor
inline __host__ __device__ double3 floor(double3 v)
{
    return make_double3(floor(v.x), floor(v.y), floor(v.z));
}

// Equal
inline __host__ __device__ bool operator==(double3 a, double3 b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

inline __host__ __device__ bool operator==(int3 a, int3 b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

// Double to Int
inline __host__ __device__ int3 double3ToInt3(double3 v)
{
    return make_int3((int)(v.x), (int)(v.y), (int)(v.z));
}

// Addition
inline __host__ __device__ int3 operator+(int3 a, int3 b)
{
    return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}