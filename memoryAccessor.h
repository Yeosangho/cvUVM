#include <cuda_runtime_api.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef memoryAccessor_h
#define memoryAccessor_h
void cuSetDeviceFlags();
void cuMallocManaged(float** h_img, int r, int c);
void cuMalloc(void** h_img, int r, int c);
void cuDeviceSynchronize();
#endif
