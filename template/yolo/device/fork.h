#ifndef FORK_H_
#define FORK_H_
#include "common.h"

#define __FORK(channel_in, channel_out, channel_from_conv, channel_to_sw, coarse_num, fine_num, rows, cols, channels, kernel_size_x, kernel_size_y, datatype) {   \
    const int coarse_index = get_compute_id(0);       \
    kernel_##datatype##_##kernel_size_x##x##kernel_size_y local_cache;      \
                                                    \
    for (uint pixel_index = 0; pixel_index < rows * cols * channels / coarse_num; ++pixel_index) { \
        write_channel_intel(channel_to_sw[coarse_index], 0);       \
        mem_fence(CLK_CHANNEL_MEM_FENCE);       \
        local_cache = read_channel_intel(channel_in[coarse_index]);   \
        _Pragma("unroll")   \
        for (uint out_index = 0; out_index < fine_num; ++out_index) { \
            read_channel_intel(channel_from_conv[coarse_index][out_index]);   \
            mem_fence(CLK_CHANNEL_MEM_FENCE);       \
            write_channel_intel(channel_out[coarse_index][out_index], local_cache);  \
        }       \
    }       \
}

/**
    @brief 复制从sliding window送来的图片窗口
    @param layer kernel函数名称
    @param channel_in 一维输入通道数组
    @param channel_out 二维输出通道数组
    @param channel_from_conv 用于与conv通信的二维通道数组
    @param channel_to_sw 用于与sliding window通信的一维通道数组
    @param coarse_num 粗粒度并行个数
    @param fine_num 细粒度并行个数
    @param rows 输出图片的高像素个数
    @param cols 输出图片的宽像素个数
    @param channels 输入图片的总通道个数
    @param kernel_size_x 卷积核高度
    @param kernel_size_y 卷积核宽度
    @param datatype 图片单个像素点的数据类型
 */
#define GEN_FORK(layer, channel_in, channel_out, channel_from_conv, channel_to_sw, coarse_num, fine_num, rows, cols, channels, kernel_size_x, kernel_size_y, datatype) \
__kernel        \
__attribute__((max_global_work_dim(0)))     \
__attribute__((autorun))    \
__attribute__((num_compute_units(coarse_num)))   \
void fork_##layer() {     \
    __FORK(channel_in, channel_out, channel_from_conv, channel_to_sw, coarse_num, fine_num, rows, cols, channels, kernel_size_x, kernel_size_y, datatype) \
}



#define __FORK_MERGE(input_width, channel_in, channel_out, rows, cols, channels, datatype) {   \
    for (uint pixel_index = 0; pixel_index < rows * cols * channels; pixel_index += input_width) { \
        _Pragma("unroll")   \
        for (uint input_idx = 0; input_idx < input_width; ++input_idx) {        \
            datatype pixel = read_channel_intel(channel_in[input_idx]);   \
            write_channel_intel(channel_out[0][input_idx], pixel);  \
            write_channel_intel(channel_out[1][input_idx], pixel);  \
        } \
    }       \
}

/**
    @brief 将一个通道的数据复制为两份并输出至两个通道
    @param layer kernel函数名称
    @param channel_in 输入通道
    @param channel_out 输出通道
    @param rows 输出图片的高像素个数
    @param cols 输出图片的宽像素个数
    @param channels 输出图片的总通道个数
    @param datatype 通道中的元素数据类型
 */
#define GEN_FORK_MERGE(input_width, layer, channel_in, channel_out, rows, cols, channels, datatype) \
__kernel        \
__attribute__((max_global_work_dim(0)))     \
__attribute__((autorun))    \
void fork_merge_##layer() {     \
    __FORK_MERGE(input_width, channel_in, channel_out, rows, cols, channels, datatype) \
}

#define __FORK_PIXEL(input_width, channel_in, channel_out_x, channel_out_y, rows, cols, channels, datatype) {   \
    for (uint pixel_index = 0; pixel_index < rows * cols * channels; pixel_index += input_width) { \
        _Pragma("unroll")   \
        for (uint input_idx = 0; input_idx < input_width; ++input_idx) {        \
            datatype pixel = read_channel_intel(channel_in[input_idx]);   \
            write_channel_intel(channel_out_x[input_idx], pixel);  \
            write_channel_intel(channel_out_y[input_idx], pixel);  \
        } \
    }       \
}

/**
    @brief 将一个通道的数据复制为两份并输出至两个通道
    @param layer kernel函数名称
    @param channel_in 输入通道
    @param channel_out_x 输出通道
    @param channel_out_y 输出通道
    @param rows 输出图片的高像素个数
    @param cols 输出图片的宽像素个数
    @param channels 输出图片的总通道个数
    @param datatype 通道中的元素数据类型
 */
#define GEN_FORK_PIXEL(input_width, layer, channel_in, channel_out_x, channel_out_y, rows, cols, channels, datatype) \
__kernel        \
__attribute__((max_global_work_dim(0)))     \
__attribute__((autorun))    \
void fork_pixel_##layer() {     \
    __FORK_PIXEL(input_width, channel_in, channel_out_x, channel_out_y, rows, cols, channels, datatype) \
}

#endif
