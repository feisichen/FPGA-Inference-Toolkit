#ifndef SLIDING_WINDOW_H_
#define SLIDING_WINDOW_H_

#define __SLIDING_WINDOW(input_width, channel_in, channel_out, zero_point_i, rows, cols, channels, pad_top, pad_rt, pad_bt, pad_lt, row_stride, col_stride, kernel_size_x, kernel_size_y, datatype)   \
    datatype frame_cache[kernel_size_x * kernel_size_y];          \
    const uint buffer_size = (kernel_size_x - 1) * (cols + pad_lt + pad_rt) * channels + (kernel_size_y - 1) * channels + 1;    \
    const uint channel_groups = channels / input_width; \
    char __attribute__((register)) buffer[buffer_size]; \
    char pixel;                    \
    for (uint row_index = 0; row_index < rows + pad_top + pad_bt; ++row_index) {    \
        for (uint col_index = 0; col_index < cols + pad_lt + pad_rt; ++col_index) {    \
            for (uint ch_group_index = 0; ch_group_index < channel_groups; ++ch_group_index) {        \
                _Pragma("unroll")   \
                for (uint ch_index = 0; ch_index < input_width; ++ch_index) {        \
                    /* pixel cache */               \
                                                    \
                    /* pixel padding */             \
                    if (row_index < pad_bt) {   \
                        pixel = zero_point_i;              \
                    } else if (row_index > rows + pad_bt - 1) { \
                        pixel= zero_point_i;              \
                    } else if (col_index < pad_lt) {  \
                        pixel = zero_point_i;              \
                    } else if (col_index > cols + pad_lt - 1) { \
                        pixel = zero_point_i;              \
                    } else {        \
                        pixel = read_channel_intel(channel_in[ch_index]);      \
                    }                                         \
                                                        \
                    _Pragma("unroll")                   \
                    for (uint i = 0; i < buffer_size - 1; ++i) {    \
                        buffer[i] = buffer[i + 1];          \
                    }       \
                    buffer[buffer_size - 1] = pixel;        \
                    /*if(coarse_index == 0)       \
                        printf("image[%d][%d][%d]: pixel=%d, buffer[%d]=%d\n", row_index, col_index, ch_index, pixel, buffer_ptr, buffer[buffer_ptr]);*/       \
                                \
                    if (row_index >= kernel_size_x - 1 && col_index >= kernel_size_y - 1 && (row_index - kernel_size_x + 1) % row_stride == 0 && (col_index - kernel_size_y + 1) % col_stride == 0) {        \
                        _Pragma("unroll")       \
                        for (uchar k1 = 0; k1 < kernel_size_x; ++k1) {     \
                            _Pragma("unroll")       \
                            for (uchar k2 = 0; k2 < kernel_size_y; ++k2) {         \
                                const uint buffer_index = k1 * channels * (cols + pad_lt + pad_rt) + k2 * channels; \
                                /*if(coarse_index == 0)   \
                                    printf("i=%d, j=%d, image_index=%d, buffer[%d]=%d\n", i, j, image_index, buffer_index, buffer[buffer_index]);*/    \
                                frame_cache[k1*kernel_size_y + k2] = buffer[buffer_index];        \
                            }           \
                        }    \
                        /*if(coarse_index == 0) { \
                            printf("[%d %d %d]:\n", row_index, col_index, ch_index);    \
                            for (uint ii = 0; ii < kernel_size_x; ++ii) {  \
                                for (uint j = 0; j < kernel_size_y; ++j) {  \
                                    printf("%d ", (int)frame_cache.data[ii][j]);  \
                                }   \
                                printf("\n");   \
                            }       \
                        }*/   \
                        /*_Pragma("unroll")   \
                        for (uint out_index = 0; out_index < fine_num; ++out_index) { \
                            read_channel_intel(channel_from_consumer[coarse_index][out_index]);   \
                        }       \
                        mem_fence(CLK_CHANNEL_MEM_FENCE);*/       \
                        _Pragma("unroll")   \
                        for(uint kernel_index = 0 ; kernel_index < kernel_size_x * kernel_size_y ; ++kernel_index) {\
                            /*printf("sw_out[%d][%d][%d][%d]:%d \n", row_index, col_index, ch_index+coarse_index, kernel_index, (int)frame_cache[coarse_index][kernel_index]);*/  \
                            write_channel_intel(channel_out[kernel_index][ch_index], frame_cache[kernel_index]); \
                        }   \
                    } \
                } \
            }           \
        }           \
    }           \

// shift register
#define __SLIDING_WINDOW_PIPE(input_width, channel_in, channel_out, zero_point_i, rows, cols, channels, pad_top, pad_rt, pad_bt, pad_lt, row_stride, col_stride, kernel_size_x, kernel_size_y, datatype)   \
    datatype frame_cache[kernel_size_x * kernel_size_y];          \
    const uint buffer_size = ((kernel_size_x - 1) * (cols + pad_lt + pad_rt) * channels + (kernel_size_y - 1) * channels + 1 ) * 2;    \
    const uint channel_groups = channels / input_width; \
    char __attribute__((register)) buffer[buffer_size]; \
    char pixel;                    \
    char origin; \
    for (uint row_index = 0; row_index < rows + pad_top + pad_bt; ++row_index) {    \
        for (uint col_index = 0; col_index < cols + pad_lt + pad_rt; ++col_index) {    \
            for (uint ch_group_index = 0; ch_group_index < channel_groups; ++ch_group_index) {        \
                _Pragma("unroll")   \
                for (uint ch_index = 0; ch_index < input_width; ++ch_index) {        \
                    /* pixel cache */               \
                                                    \
                    /* pixel padding */             \
                    if (row_index < pad_bt) {   \
                        pixel = zero_point_i;              \
                        origin = zero_point_i; \
                    } else if (row_index > rows + pad_bt - 1) { \
                        pixel = zero_point_i;              \
                        origin = zero_point_i; \
                    } else if (col_index < pad_lt) {  \
                        pixel = zero_point_i;              \
                        origin = zero_point_i; \
                    } else if (col_index > cols + pad_lt - 1) { \
                        pixel = zero_point_i;              \
                        origin = zero_point_i; \
                    } else {        \
                        pixel = read_channel_intel(channel_in[0][ch_index]);      \
                        origin = read_channel_intel(channel_in[1][ch_index]); \
                    }                                         \
                                                        \
                    _Pragma("unroll")                   \
                    for (uint i = 0; i < buffer_size - 2; ++i) {    \
                        buffer[i] = buffer[i + 2];          \
                    }       \
                    buffer[buffer_size - 2] = pixel;        \
                    buffer[buffer_size - 1] = origin;        \
                    /*if(coarse_index == 0)       \
                        printf("image[%d][%d][%d]: pixel=%d, buffer[%d]=%d\n", row_index, col_index, ch_index, pixel, buffer_ptr, buffer[buffer_ptr]);*/       \
                                \
                    if (row_index >= kernel_size_x - 1 && col_index >= kernel_size_y - 1 && (row_index - kernel_size_x + 1) % row_stride == 0 && (col_index - kernel_size_y + 1) % col_stride == 0) {        \
                        _Pragma("unroll")       \
                        for (uchar k1 = 0; k1 < kernel_size_x; ++k1) {     \
                            _Pragma("unroll")       \
                            for (uchar k2 = 0; k2 < kernel_size_y; ++k2) {         \
                                const uint buffer_index = (k1 * channels * (cols + pad_lt + pad_rt) + k2 * channels) * 2; \
                                /*if(coarse_index == 0)   \
                                    printf("i=%d, j=%d, image_index=%d, buffer[%d]=%d\n", i, j, image_index, buffer_index, buffer[buffer_index]);*/    \
                                frame_cache[k1*kernel_size_y + k2] = buffer[buffer_index];        \
                            }           \
                        }    \
                        /*if(coarse_index == 0) { \
                            printf("[%d %d %d]:\n", row_index, col_index, ch_index);    \
                            for (uint ii = 0; ii < kernel_size_x; ++ii) {  \
                                for (uint j = 0; j < kernel_size_y; ++j) {  \
                                    printf("%d ", (int)frame_cache.data[ii][j]);  \
                                }   \
                                printf("\n");   \
                            }       \
                        }*/   \
                        /*_Pragma("unroll")   \
                        for (uint out_index = 0; out_index < fine_num; ++out_index) { \
                            read_channel_intel(channel_from_consumer[coarse_index][out_index]);   \
                        }       \
                        mem_fence(CLK_CHANNEL_MEM_FENCE);*/       \
                        _Pragma("unroll")   \
                        for(uint kernel_index = 0 ; kernel_index < kernel_size_x * kernel_size_y ; ++kernel_index) {\
                            /*printf("sw_out[%d][%d][%d][%d]:%d \n", row_index, col_index, ch_index+coarse_index, kernel_index, (int)frame_cache[coarse_index][kernel_index]);*/  \
                            write_channel_intel(channel_out[kernel_index][ch_index], frame_cache[kernel_index]); \
                        }   \
                        write_channel_intel(channel_out[kernel_size_x * kernel_size_y][ch_index], buffer[((kernel_size_x / 2) * channels * (cols + pad_lt + pad_rt) + (kernel_size_y / 2) * channels) * 2 + 1]);  \
                    } \
                } \
            }           \
        }           \
    }           \

/**
    @brief 从channel读取图片，构成滑动窗口后送出
    @param name kernel函数名称
    @param channel_in 一维输入通道数组
    @param channel_out 一维输出通道数组
    @param channel_from_consumer 用于与conv通信的一维通道数组
    @param coarse_num 粗粒度并行个数
    @param zero_point_i 本层的原点, padding时填充的值
    @param rows 输入图片的高像素个数
    @param cols 输入图片的宽像素个数
    @param channels 输入图片的总通道个数
    @param pad_top 图片上边缘填充个数
    @param pad_rt 图片右边缘填充个数
    @param pad_bt 图片下边缘填充个数
    @param pad_lt 图片左边缘填充个数
    @param row_stride 行步进
    @param col_stride 列步进
    @param kernel_size_x 卷积核高度
    @param kernel_size_y 卷积核宽度
    @param datatype 图片单个像素点的数据类型
 */

#define GEN_SLIDING_WINDOW(input_width, name, channel_in, channel_out, zero_point, rows, cols, channels, pad_top, pad_rt, pad_bt, pad_lt, row_stride, col_stride, kernel_size_x, kernel_size_y, datatype) \
__kernel        \
__attribute__((max_global_work_dim(0)))     \
__attribute__((autorun))    \
/*__attribute__((num_compute_units(coarse_num)))*/   \
void sliding_window_##name() {     \
    __SLIDING_WINDOW(input_width, channel_in, channel_out, zero_point, rows, cols, channels, pad_top, pad_rt, pad_bt, pad_lt, row_stride, col_stride, kernel_size_x, kernel_size_y, datatype) \
}

#define GEN_SLIDING_WINDOW_PIPE(input_width, name, channel_in, channel_out, zero_point, rows, cols, channels, pad_top, pad_rt, pad_bt, pad_lt, row_stride, col_stride, kernel_size_x, kernel_size_y, datatype) \
__kernel        \
__attribute__((max_global_work_dim(0)))     \
__attribute__((autorun))    \
/*__attribute__((num_compute_units(coarse_num)))*/   \
void sliding_window_##name() {     \
    __SLIDING_WINDOW_PIPE(input_width, channel_in, channel_out, zero_point, rows, cols, channels, pad_top, pad_rt, pad_bt, pad_lt, row_stride, col_stride, kernel_size_x, kernel_size_y, datatype) \
}

#endif
