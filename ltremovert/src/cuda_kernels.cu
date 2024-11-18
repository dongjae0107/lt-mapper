#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cmath>
#include <limits>

// Utility functions
__device__ float rad2deg(float radians) {
    return radians * 180.0f
}

__device__ void cart2sph(float x, float y, float z, float &r, float &az, float &el) {
    r = sqrtf(x * x + y * y + z * z);
    az = atan2f(y, x);
    el = asinf(z / r);
}

// CUDA Kernel
__global__ void scanToRangeImageKernel(const float *x, const float *y, const float *z,
                                       float *rimg, int num_points,
                                       float vfov, float hfov,
                                       int rimg_rows, int rimg_cols,
                                       float flag_no_point) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    // Extract the point
    float px = x[idx];
    float py = y[idx];
    float pz = z[idx];

    // Convert Cartesian to spherical
    float r, az, el;
    cart2sph(px, py, pz, r, az, el);

    // Map to range image coordinates
    int row = __float2int_rn(fminf(fmaxf(rimg_rows * (1 - (rad2deg(el) + vfov / 2.0f) / vfov), 0.0f), rimg_rows - 1));
    int col = __float2int_rn(fminf(fmaxf(rimg_cols * ((rad2deg(az) + hfov / 2.0f) / hfov), 0.0f), rimg_cols - 1));

    // Atomic update for range image
    int pixel_idx = row * rimg_cols + col;
    atomicMin(&rimg[pixel_idx], r);
}

// Host function exposed for C++ integration
extern "C" void scanToRangeImageCuda(const float *x, const float *y, const float *z,
                                  float *rimg, int num_points,
                                  float vfov, float hfov,
                                  int rimg_rows, int rimg_cols,
                                  float flag_no_point) {
    // Allocate device memory
    float *d_x, *d_y, *d_z, *d_rimg;
    size_t points_size = num_points * sizeof(float);
    size_t rimg_size = rimg_rows * rimg_cols * sizeof(float);
    cudaMalloc(&d_x, points_size);
    cudaMalloc(&d_y, points_size);
    cudaMalloc(&d_z, points_size);
    cudaMalloc(&d_rimg, rimg_size);

    // Copy data to device
    cudaMemcpy(d_x, x, points_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, points_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, z, points_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rimg, rimg, rimg_size, cudaMemcpyHostToDevice);

    // Launch kernel
    int threads_per_block = 256;
    int blocks = (num_points + threads_per_block - 1) / threads_per_block;
    scanToRangeImageKernel<<<blocks, threads_per_block>>>(d_x, d_y, d_z, d_rimg,
                                                          num_points, vfov, hfov,
                                                          rimg_rows, rimg_cols,
                                                          flag_no_point);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(rimg, d_rimg, rimg_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_rimg);
}