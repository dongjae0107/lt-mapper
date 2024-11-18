#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

extern "C" void scanToRangeImageCuda(const float *x, const float *y, const float *z,
                                        float *rimg, int num_points,
                                        float vfov, float hfov,
                                        int rimg_rows, int rimg_cols,
                                        float flag_no_point);


#endif // CUDA_KERNELS_H