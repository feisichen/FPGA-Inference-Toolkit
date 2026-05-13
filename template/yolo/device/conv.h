#ifndef CONV_H_
#define CONV_H_
#include "common.h"
#include "rtl_lib.h"

#define __CONV_FINE_CAL(input_width, output_width, coarse_num, channel_in_img, channel_out, zero_point_i, sigmoid, bias_conv, Num_conv, N_conv, zero_point_o, zero_point_i_sig, zero_point_o_mul, Num_mul, N_mul, weights, rows, cols, channels, filters, kernel_size_x, kernel_size_y, num_banks, conv_data_type, weight_data_type, conv_acc_type) \
    const uint kernel_size_total = kernel_size_x * kernel_size_y; \
    const uint weight_num = channels * filters / coarse_num ;     \
    conv_acc_type cache[filters]; \
    conv_acc_type __attribute__((register)) sum[coarse_num]; \
    conv_data_type __attribute__((register)) local_cache[kernel_size_total][input_width];    \
    weight_data_type __attribute__((numbanks(num_banks), bankwidth(1))) weight_internal[weight_num][num_banks];      \
                                                            \
    /*for(uint i = 0; i < WINDOW_CHANNEL_DEPTH; ++i) {  \
        write_channel_intel(channel_to_sw[coarse_index][fine_index], 0);    \
    }*/       \
    \
    for(uint channel_index = 0; channel_index < channels; ++channel_index) { \
        for (uint filter_index = 0; filter_index < (filters / coarse_num); ++filter_index) { \
            for (uint kernel_index = 0; kernel_index < kernel_size_total ; ++kernel_index) { \
                for(uint coarse_index=0 ; coarse_index < coarse_num ; ++coarse_index){ \
                    uint weights_index = ((filter_index * coarse_num + coarse_index)  * channels + channel_index) * kernel_size_total + kernel_index; \
                    weight_internal[channel_index * (filters / coarse_num) + filter_index][kernel_index * coarse_num + coarse_index] = weights[weights_index];         \
                } \
            } \
        }       \
    } \
    while(1) {  \
        for(uint i = 0; i < rows * cols; i++) { \
            _Pragma("unroll")      \
            for (uint j = 0; j < filters; j++) { \
                cache[j] = 0; \
            } \
            for(uint weight_internal_index = 0, channel_group_index = 0; channel_group_index < channels; channel_group_index += input_width) { \
                _Pragma("unroll")   \
                for(uint input_idx = 0; input_idx < input_width; ++input_idx){ \
                    _Pragma("unroll")   \
                    for(uint kernel_index = 0 ; kernel_index < kernel_size_total ; ++kernel_index) { \
                        local_cache[kernel_index][input_idx] = read_channel_intel(channel_in_img[kernel_index][input_idx]); \
                    }   \
                } \
                for(uint input_idx = 0; input_idx < input_width; ++input_idx){ \
                    for (uint filter_index = 0; filter_index < (filters / coarse_num); ++filter_index) { \
                        _Pragma("unroll")   \
                        for(uint coarse_index=0 ; coarse_index < coarse_num ; ++coarse_index){ \
                            sum[coarse_index] = 0; \
                        } \
                        _Pragma("unroll")   \
                        for (uint kernel_index = 0; kernel_index < kernel_size_total ; ++kernel_index) { \
                            _Pragma("unroll")   \
                            for(uint coarse_index=0 ; coarse_index < coarse_num ; ++coarse_index){ \
                                /*printf("mul[%d][%d][%d][%d]:%d, %d \n", i, filter_index + fine_index, channel_index + coarse_index, kernel_index, (int) (local_cache[bank_index] - zero_point_i), (int)weight_internal[weight_internal_index][bank_index]);*/  \
                                sum[coarse_index] += (local_cache[kernel_index][input_idx] - zero_point_i) * weight_internal[(channel_group_index + input_idx) * (filters / coarse_num) + filter_index][kernel_index * coarse_num + coarse_index]; \
                            } \
                        } \
                        _Pragma("unroll")   \
                        for(uint coarse_index=0 ; coarse_index < coarse_num ; ++coarse_index){ \
                            sum[coarse_index] += cache[coarse_index];  \
                        } \
                        _Pragma("unroll")   \
                        for (uint j = 0; j < filters - coarse_num; j++) { \
                            cache[j] = cache[j + coarse_num];    \
                        } \
                        _Pragma("unroll")   \
                        for(uint coarse_index=0 ; coarse_index < coarse_num ; ++coarse_index){ \
                            cache[filters - coarse_num + coarse_index] = sum[coarse_index]; \
                        } \
                    } \
                } \
            } \
            for (uint filter_index = 0; filter_index < filters; filter_index += output_width) { \
                _Pragma("unroll")   \
                for(uint output_idx = 0; output_idx < output_width; ++output_idx){ \
                    conv_acc_type temp = rightShiftWithRound((bias_conv[filter_index + output_idx] + cache[output_idx]) * Num_conv[filter_index + output_idx], N_conv[filter_index + output_idx]); \
                    conv_data_type glue_out = mul(temp + zero_point_o, sigmoid(temp), zero_point_o, zero_point_i_sig, zero_point_o_mul, Num_mul, N_mul); \
                    write_channel_intel(channel_out[output_idx], glue_out);  \
                } \
                _Pragma("unroll")   \
                for (uint j = 0; j < filters - output_width; j++) { \
                    cache[j] = cache[j + output_width];    \
                } \
            } \
        }   \
    } \

#define __CONV_PIPE_CAL(input_width, output_width, coarse_num, channel_in_img, channel_out, zero_point_i, sigmoid, bias_conv, Num_conv, N_conv, zero_point_o, zero_point_i_sig, zero_point_o_mul, Num_mul, N_mul, weights, rows, cols, channels, filters, kernel_size_x, kernel_size_y, num_banks, conv_data_type, weight_data_type, conv_acc_type) \
    const uint kernel_size_total = kernel_size_x * kernel_size_y; \
    const uint weight_num = (channels * filters / coarse_num) / 2;     \
    conv_acc_type cache[filters]; \
    conv_acc_type origin[filters]; \
    conv_acc_type __attribute__((register)) sum[coarse_num]; \
    conv_data_type __attribute__((register)) local_cache[kernel_size_total * 2][input_width];    \
    weight_data_type __attribute__((numbanks(num_banks * 2), bankwidth(1))) weight_internal[weight_num][num_banks * 2];      \
                                                            \
    /*for(uint i = 0; i < WINDOW_CHANNEL_DEPTH; ++i) {  \
        write_channel_intel(channel_to_sw[coarse_index][fine_index], 0);    \
    }*/       \
    \
    for(uint channel_group_index = 0; channel_group_index < channels / 2; ++channel_group_index) { \
        for (uint filter_index = 0; filter_index < (filters / coarse_num); ++filter_index) { \
            for (uint kernel_index = 0; kernel_index < kernel_size_total ; ++kernel_index) { \
                for(uint coarse_index=0 ; coarse_index < coarse_num ; ++coarse_index){ \
                    for(uint channel_index = 0; channel_index < 2; ++channel_index) { \
                        uint weights_index = ((filter_index * coarse_num + coarse_index)  * channels + channel_index + channel_group_index * 2) * kernel_size_total + kernel_index; \
                        weight_internal[channel_group_index * (filters / coarse_num) + filter_index][(kernel_index * coarse_num + coarse_index) * 2 + channel_index] = weights[weights_index];         \
                    } \
                } \
            } \
        }       \
    } \
    while(1) {  \
        for(uint i = 0; i < rows * cols; i++) { \
            _Pragma("unroll")      \
            for (uint j = 0; j < filters; j++) { \
                cache[j] = 0; \
                origin[j] = 0; \
            } \
            for(uint channel_group_index = 0; channel_group_index < channels / 2 ; channel_group_index += input_width) { \
                for(uint channel_index = 0; channel_index < 2; ++channel_index) { \
                    _Pragma("unroll")   \
                    for (uint j = 0; j < filters - input_width; j++) { \
                        origin[j] = origin[j + input_width];    \
                    } \
                    _Pragma("unroll")   \
                    for(uint input_idx = 0; input_idx < input_width; ++input_idx){ \
                        _Pragma("unroll")   \
                        for(uint kernel_index = 0 ; kernel_index < kernel_size_total ; ++kernel_index) { \
                            local_cache[kernel_index * 2 + channel_index][input_idx] = read_channel_intel(channel_in_img[kernel_index][input_idx]); \
                        } \
                        origin[filters - 1] = read_channel_intel(channel_in_img[kernel_size_total][input_idx]); \
                    } \
                }   \
                for(uint input_idx = 0; input_idx < input_width; ++input_idx){ \
                    for (uint filter_index = 0; filter_index < (filters / coarse_num); ++filter_index) { \
                        _Pragma("unroll")   \
                        for(uint coarse_index=0 ; coarse_index < coarse_num ; ++coarse_index){ \
                            sum[coarse_index] = 0; \
                        } \
                        _Pragma("unroll")   \
                        for (uint kernel_index = 0; kernel_index < kernel_size_total ; ++kernel_index) { \
                            _Pragma("unroll")   \
                            for(uint coarse_index=0 ; coarse_index < coarse_num ; ++coarse_index){ \
                                int8_t l0 = kernel_index * 2; \
                                int8_t l1 = kernel_index * 2 + 1; \
                                uint w = (channel_group_index + input_idx) * (filters / coarse_num) + filter_index; \
                                weight_data_type w0 = weight_internal[w][(kernel_index * coarse_num + coarse_index) * 2]; \
                                weight_data_type w1 = weight_internal[w][(kernel_index * coarse_num + coarse_index) * 2 + 1]; \
                                sum[coarse_index] += mult_add_fix8bx4(local_cache[l0][input_idx], w0, zero_point_i, -w0, local_cache[l1][input_idx], w1, zero_point_i, -w1); \
                            } \
                        } \
                        _Pragma("unroll")   \
                        for(uint coarse_index=0 ; coarse_index < coarse_num ; ++coarse_index){ \
                            sum[coarse_index] += cache[coarse_index];  \
                        } \
                        _Pragma("unroll")   \
                        for (uint j = 0; j < filters - coarse_num; j++) { \
                            cache[j] = cache[j + coarse_num];    \
                        } \
                        _Pragma("unroll")   \
                        for(uint coarse_index=0 ; coarse_index < coarse_num ; ++coarse_index){ \
                            cache[filters - coarse_num + coarse_index] = sum[coarse_index]; \
                        } \
                    } \
                } \
            } \
            for (uint filter_index = 0; filter_index < filters; filter_index += output_width) { \
                _Pragma("unroll")   \
                for(uint output_idx = 0; output_idx < output_width; ++output_idx){ \
                    conv_acc_type temp = rightShiftWithRound((bias_conv[filter_index + output_idx] + cache[output_idx]) * Num_conv[filter_index + output_idx], N_conv[filter_index + output_idx]); \
                    conv_data_type glue_out = mul(temp + zero_point_o, sigmoid(temp), zero_point_o, zero_point_i_sig, zero_point_o_mul, Num_mul, N_mul); \
                    write_channel_intel(channel_out[0][output_idx], glue_out);  \
                    write_channel_intel(channel_out[1][output_idx], origin[output_idx]);  \
                } \
                _Pragma("unroll")   \
                for (uint j = 0; j < filters - output_width; j++) { \
                    cache[j] = cache[j + output_width];    \
                    origin[j] = origin[j + output_width];    \
                } \
            } \
        }   \
    } \

#define __CONV_CAL(input_width, output_width, coarse_num, channel_in_img, channel_out, zero_point_i, sigmoid, bias_conv, Num_conv, N_conv, zero_point_o, zero_point_i_sig, zero_point_o_mul, Num_mul, N_mul, weights, rows, cols, channels, filters, kernel_size_x, kernel_size_y, num_banks, conv_data_type, weight_data_type, conv_acc_type) \
    const uint kernel_size_total = kernel_size_x * kernel_size_y; \
    const uint weight_num = (channels * filters / coarse_num) / 2;     \
    conv_acc_type cache[filters]; \
    conv_acc_type __attribute__((register)) sum[coarse_num]; \
    conv_data_type __attribute__((register)) local_cache[kernel_size_total * 2][input_width];    \
    weight_data_type __attribute__((numbanks(num_banks * 2), bankwidth(1))) weight_internal[weight_num][num_banks * 2];      \
                                                            \
    /*for(uint i = 0; i < WINDOW_CHANNEL_DEPTH; ++i) {  \
        write_channel_intel(channel_to_sw[coarse_index][fine_index], 0);    \
    }*/       \
    \
    for(uint channel_group_index = 0; channel_group_index < channels / 2; ++channel_group_index) { \
        for (uint filter_index = 0; filter_index < (filters / coarse_num); ++filter_index) { \
            for (uint kernel_index = 0; kernel_index < kernel_size_total ; ++kernel_index) { \
                for(uint coarse_index=0 ; coarse_index < coarse_num ; ++coarse_index){ \
                    for(uint channel_index = 0; channel_index < 2; ++channel_index) { \
                            uint weights_index = ((filter_index * coarse_num + coarse_index)  * channels + channel_index + channel_group_index * 2) * kernel_size_total + kernel_index; \
                            weight_internal[channel_group_index * (filters / coarse_num) + filter_index][(kernel_index * coarse_num + coarse_index) * 2 + channel_index] = weights[weights_index];         \
                    } \
                } \
            } \
        }       \
    } \
    while(1) {  \
        for(uint i = 0; i < rows * cols; i++) { \
            _Pragma("unroll")      \
            for (uint j = 0; j < filters; j++) { \
                cache[j] = 0; \
            } \
            for(uint channel_group_index = 0; channel_group_index < channels / 2 ; channel_group_index += input_width) { \
                for(uint channel_index = 0; channel_index < 2; ++channel_index) { \
                    _Pragma("unroll")   \
                    for(uint input_idx = 0; input_idx < input_width; ++input_idx){ \
                        _Pragma("unroll")   \
                        for(uint kernel_index = 0 ; kernel_index < kernel_size_total ; ++kernel_index) { \
                            local_cache[kernel_index * 2 + channel_index][input_idx] = read_channel_intel(channel_in_img[kernel_index][input_idx]); \
                        } \
                    } \
                }   \
                for(uint input_idx = 0; input_idx < input_width; ++input_idx){ \
                    for (uint filter_index = 0; filter_index < (filters / coarse_num); ++filter_index) { \
                        _Pragma("unroll")   \
                        for(uint coarse_index=0 ; coarse_index < coarse_num ; ++coarse_index){ \
                            sum[coarse_index] = 0; \
                        } \
                        _Pragma("unroll")   \
                        for (uint kernel_index = 0; kernel_index < kernel_size_total ; ++kernel_index) { \
                            _Pragma("unroll")   \
                            for(uint coarse_index=0 ; coarse_index < coarse_num ; ++coarse_index){ \
                                int8_t l0 = kernel_index * 2; \
                                int8_t l1 = kernel_index * 2 + 1; \
                                uint w = (channel_group_index + input_idx) * (filters / coarse_num) + filter_index; \
                                weight_data_type w0 = weight_internal[w][(kernel_index * coarse_num + coarse_index) * 2]; \
                                weight_data_type w1 = weight_internal[w][(kernel_index * coarse_num + coarse_index) * 2 + 1]; \
                                sum[coarse_index] += mult_add_fix8bx4(local_cache[l0][input_idx], w0, zero_point_i, -w0, local_cache[l1][input_idx], w1, zero_point_i, -w1); \
                            } \
                        } \
                        _Pragma("unroll")   \
                        for(uint coarse_index=0 ; coarse_index < coarse_num ; ++coarse_index){ \
                            sum[coarse_index] += cache[coarse_index];  \
                        } \
                        _Pragma("unroll")   \
                        for (uint j = 0; j < filters - coarse_num; j++) { \
                            cache[j] = cache[j + coarse_num];    \
                        } \
                        _Pragma("unroll")   \
                        for(uint coarse_index=0 ; coarse_index < coarse_num ; ++coarse_index){ \
                            cache[filters - coarse_num + coarse_index] = sum[coarse_index]; \
                        } \
                    } \
                } \
            } \
            for (uint filter_index = 0; filter_index < filters; filter_index += output_width) { \
                _Pragma("unroll")   \
                for(uint output_idx = 0; output_idx < output_width; ++output_idx){ \
                    conv_acc_type temp = rightShiftWithRound((bias_conv[filter_index + output_idx] + cache[output_idx]) * Num_conv[filter_index + output_idx], N_conv[filter_index + output_idx]); \
                    conv_data_type glue_out = mul(temp + zero_point_o, sigmoid(temp), zero_point_o, zero_point_i_sig, zero_point_o_mul, Num_mul, N_mul); \
                    write_channel_intel(channel_out[output_idx], glue_out);  \
                } \
                _Pragma("unroll")   \
                for (uint j = 0; j < filters - output_width; j++) { \
                    cache[j] = cache[j + output_width];    \
                } \
            } \
        }   \
    } \

/**
    @brief 卷积计算
 */
#define GEN_CONV_CAL(input_width, output_width, coarse_num, layer, channel_in_img, channel_out, zero_point_i, sigmoid, bias_conv, Num_conv, N_conv, zero_point_o, zero_point_i_sig, zero_point_o_mul, Num_mul, N_mul, weights, rows, cols, channels, filters, kernel_size_x, kernel_size_y, num_banks, conv_data_type, weight_data_type, conv_acc_type) \
__kernel        \
__attribute__((max_global_work_dim(0)))     \
__attribute__((autorun))    \
/*__attribute__((num_compute_units(coarse_num, fine_num)))*/   \
void conv_cal_##layer() {     \
    __CONV_CAL(input_width, output_width, coarse_num, channel_in_img, channel_out, zero_point_i, sigmoid, bias_conv, Num_conv, N_conv, zero_point_o, zero_point_i_sig, zero_point_o_mul, Num_mul, N_mul, weights, rows, cols, channels, filters, kernel_size_x, kernel_size_y, num_banks, conv_data_type, weight_data_type, conv_acc_type) \
}

#define GEN_PIPE_CONV_CAL(input_width, output_width, coarse_num, layer, channel_in_img, channel_out, zero_point_i, sigmoid, bias_conv, Num_conv, N_conv, zero_point_o, zero_point_i_sig, zero_point_o_mul, Num_mul, N_mul, weights, rows, cols, channels, filters, kernel_size_x, kernel_size_y, num_banks, conv_data_type, weight_data_type, conv_acc_type) \
__kernel        \
__attribute__((max_global_work_dim(0)))     \
__attribute__((autorun))    \
/*__attribute__((num_compute_units(coarse_num, fine_num)))*/   \
void conv_pipe_cal_##layer() {     \
    __CONV_PIPE_CAL(input_width, output_width, coarse_num, channel_in_img, channel_out, zero_point_i, sigmoid, bias_conv, Num_conv, N_conv, zero_point_o, zero_point_i_sig, zero_point_o_mul, Num_mul, N_mul, weights, rows, cols, channels, filters, kernel_size_x, kernel_size_y, num_banks, conv_data_type, weight_data_type, conv_acc_type) \
}

#define GEN_FINE_CONV_CAL(input_width, output_width, coarse_num, layer, channel_in_img, channel_out, zero_point_i, sigmoid, bias_conv, Num_conv, N_conv, zero_point_o, zero_point_i_sig, zero_point_o_mul, Num_mul, N_mul, weights, rows, cols, channels, filters, kernel_size_x, kernel_size_y, num_banks, conv_data_type, weight_data_type, conv_acc_type) \
__kernel        \
__attribute__((max_global_work_dim(0)))     \
__attribute__((autorun))    \
/*__attribute__((num_compute_units(coarse_num, fine_num)))*/   \
void conv_fine_cal_##layer() {     \
    __CONV_FINE_CAL(input_width, output_width, coarse_num, channel_in_img, channel_out, zero_point_i, sigmoid, bias_conv, Num_conv, N_conv, zero_point_o, zero_point_i_sig, zero_point_o_mul, Num_mul, N_mul, weights, rows, cols, channels, filters, kernel_size_x, kernel_size_y, num_banks, conv_data_type, weight_data_type, conv_acc_type) \
}
/**
    @brief 对送来的图片窗口进行卷积操作
    @param layer kernel函数名称
    @param channel_in 二维输入通道数组
    @param channel_out 二维输出通道数组
    @param channel_to_sw 用于与sw通信的二维通道数组
    @param coarse_num 粗粒度并行个数
    @param fine_num 细粒度并行个数
    @param zero_point_i 本层的图片原点
    @param weights 本层的一维权重数组
    @param rows 输出图片的高像素个数
    @param cols 输出图片的宽像素个数
    @param channels 输入图片的总通道个数
    @param filters 输出图片的总通道个数（过滤器数量）
    @param kernel_size_x 卷积核高度
    @param kernel_size_y 卷积核宽度
    @param num_banks 权重缓冲区的memory bank个数
    @param conv_data_type 图片的元素数据类型
    @param weight_data_type 权重的数据类型
    @param conv_acc_type 卷积结果的数据类型
    @note 细粒度并行度大于1时，每个conv负责计算的输出通道交叉分布
 */
#define GEN_CONV(input_width, output_width, coarse_num, layer, channel_in, channel_out, zero_point_i, sigmoid, bias_conv, Num_conv, N_conv, zero_point_o, zero_point_i_sig, zero_point_o_mul, Num_mul, N_mul, weights, rows, cols, channels, filters, kernel_size_x, kernel_size_y, num_banks, conv_data_type, weight_data_type, conv_acc_type)    \
/*channel int8_t conv_cal_wt_com_##layer[coarse_num][fine_num] __attribute__((depth(0)));     \
channel KERNEL_TYPE(kernel_size_x, kernel_size_y, weight_data_type) conv_wt_out_##layer[coarse_num][fine_num] __attribute__((depth(0)));     \
GEN_CONV_LD_WT(layer, conv_wt_out_##layer, conv_cal_wt_com_##layer, coarse_num, fine_num, weights, rows, cols, channels, filters, kernel_size_x, kernel_size_y, num_banks, weight_data_type)*/  \
GEN_CONV_CAL(input_width, output_width, coarse_num, layer, channel_in, channel_out, zero_point_i, sigmoid, bias_conv, Num_conv, N_conv, zero_point_o, zero_point_i_sig, zero_point_o_mul, Num_mul, N_mul, weights, rows, cols, channels, filters, kernel_size_x, kernel_size_y, num_banks, conv_data_type, weight_data_type, conv_acc_type)

#define GEN_PIPE_CONV(input_width, output_width, coarse_num, layer, channel_in, channel_out, zero_point_i, sigmoid, bias_conv, Num_conv, N_conv, zero_point_o, zero_point_i_sig, zero_point_o_mul, Num_mul, N_mul, weights, rows, cols, channels, filters, kernel_size_x, kernel_size_y, num_banks, conv_data_type, weight_data_type, conv_acc_type)    \
/*channel int8_t conv_cal_wt_com_##layer[coarse_num][fine_num] __attribute__((depth(0)));     \
channel KERNEL_TYPE(kernel_size_x, kernel_size_y, weight_data_type) conv_wt_out_##layer[coarse_num][fine_num] __attribute__((depth(0)));     \
GEN_CONV_LD_WT(layer, conv_wt_out_##layer, conv_cal_wt_com_##layer, coarse_num, fine_num, weights, rows, cols, channels, filters, kernel_size_x, kernel_size_y, num_banks, weight_data_type)*/  \
GEN_PIPE_CONV_CAL(input_width, output_width, coarse_num, layer, channel_in, channel_out, zero_point_i, sigmoid, bias_conv, Num_conv, N_conv, zero_point_o, zero_point_i_sig, zero_point_o_mul, Num_mul, N_mul, weights, rows, cols, channels, filters, kernel_size_x, kernel_size_y, num_banks, conv_data_type, weight_data_type, conv_acc_type)

#define GEN_FINE_CONV(input_width, output_width, coarse_num, layer, channel_in, channel_out, zero_point_i, sigmoid, bias_conv, Num_conv, N_conv, zero_point_o, zero_point_i_sig, zero_point_o_mul, Num_mul, N_mul, weights, rows, cols, channels, filters, kernel_size_x, kernel_size_y, num_banks, conv_data_type, weight_data_type, conv_acc_type)    \
/*channel int8_t conv_cal_wt_com_##layer[coarse_num][fine_num] __attribute__((depth(0)));     \
channel KERNEL_TYPE(kernel_size_x, kernel_size_y, weight_data_type) conv_wt_out_##layer[coarse_num][fine_num] __attribute__((depth(0)));     \
GEN_CONV_LD_WT(layer, conv_wt_out_##layer, conv_cal_wt_com_##layer, coarse_num, fine_num, weights, rows, cols, channels, filters, kernel_size_x, kernel_size_y, num_banks, weight_data_type)*/  \
GEN_FINE_CONV_CAL(input_width, output_width, coarse_num, layer, channel_in, channel_out, zero_point_i, sigmoid, bias_conv, Num_conv, N_conv, zero_point_o, zero_point_i_sig, zero_point_o_mul, Num_mul, N_mul, weights, rows, cols, channels, filters, kernel_size_x, kernel_size_y, num_banks, conv_data_type, weight_data_type, conv_acc_type)

#endif