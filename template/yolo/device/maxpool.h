#ifndef MAXPOOL_H_
#define MAXPOOL_H_
#include "type.h"
#include "sliding_window.h"

#define CHANNELS_MAXPOOL_0 128
#define HEIGHT_MAXPOOL_0 20
#define WIDTH_MAXPOOL_0 20
#define KERNEL_SIZE_MAXPOOL_0 5
#define PAD_MAXPOOL_0 2
#define STRIDE_MAXPOOL_0 1
#define ZERO_POINT_MAXPOOL_0 -119

#define CHANNELS_MAXPOOL_1 128
#define HEIGHT_MAXPOOL_1 20
#define WIDTH_MAXPOOL_1 20
#define KERNEL_SIZE_MAXPOOL_1 5
#define PAD_MAXPOOL_1 2
#define STRIDE_MAXPOOL_1 1
#define ZERO_POINT_MAXPOOL_1 -119

#define CHANNELS_MAXPOOL_2 128
#define HEIGHT_MAXPOOL_2 20
#define WIDTH_MAXPOOL_2 20
#define KERNEL_SIZE_MAXPOOL_2 5
#define PAD_MAXPOOL_2 2
#define STRIDE_MAXPOOL_2 1
#define ZERO_POINT_MAXPOOL_2 -119

#define __MAXPOOL(channel_in, channel_out, rows, cols, channels, kernel_size_x, kernel_size_y, datatype) {    \
    const uint kernel_size_total = kernel_size_x * kernel_size_y; \
    datatype __attribute__((register)) local_cache[kernel_size_total];    \
    datatype max;            \
    for (uint pixel_index = 0; pixel_index < rows * cols * channels; ++pixel_index) { \
        _Pragma("unroll")   \
        for(uint kernel_index = 0 ; kernel_index < kernel_size_total ; ++kernel_index) { \
            local_cache[kernel_index] = read_channel_intel(channel_in[kernel_index]); \
        }   \
        max = local_cache[0];         \
        _Pragma("unroll")   \
        for(uint kernel_index = 1 ; kernel_index < kernel_size_total ; ++kernel_index) { \
            max = local_cache[kernel_index] > max ? local_cache[kernel_index] : max;   \
        }   \
        write_channel_intel(channel_out, max);    \
    }     \
}

#define _GEN_MAXPOOL(index, channel_in, channel_out, rows, cols, channels, kernel_size_x, kernel_size_y, datatype)  \
__kernel        \
__attribute__((max_global_work_dim(0)))     \
__attribute__((autorun))    \
void maxpool_##index() {     \
    __MAXPOOL(channel_in, channel_out, rows, cols, channels, kernel_size_x, kernel_size_y, datatype) \
}

#define GEN_MAXPOOL(index, channel_in, channel_out, kernel_size, datatype)   \
channel datatype sw_out_maxpool_##index[HEIGHT_MAXPOOL_##index *  WIDTH_MAXPOOL_##index] __attribute__((depth(1)));     \
GEN_SLIDING_WINDOW(maxpool##index, channel_in, sw_out_maxpool_##index, ZERO_POINT_MAXPOOL_##index, HEIGHT_MAXPOOL_##index, WIDTH_MAXPOOL_##index, CHANNELS_MAXPOOL_##index, PAD_MAXPOOL_##index, PAD_MAXPOOL_##index, PAD_MAXPOOL_##index, PAD_MAXPOOL_##index, STRIDE_MAXPOOL_##index, STRIDE_MAXPOOL_##index, KERNEL_SIZE_MAXPOOL_##index, KERNEL_SIZE_MAXPOOL_##index, datatype)   \
_GEN_MAXPOOL(index, sw_out_maxpool_##index, channel_out, HEIGHT_MAXPOOL_##index, WIDTH_MAXPOOL_##index, CHANNELS_MAXPOOL_##index, KERNEL_SIZE_MAXPOOL_##index, KERNEL_SIZE_MAXPOOL_##index, datatype)

#endif