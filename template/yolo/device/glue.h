#ifndef GLUE_H_
#define GLUE_H_

#define __GLUE(channel_in, channel_out, coarse_num, fine_num, sigmoid, bias_conv, Num_conv, N_conv, zero_point_o, zero_point_i_sig, zero_point_o_mul, Num_mul, N_mul, rows, cols, filters, data_type, glue_type) \
    const uint filters_per_group = DIVIDE(filters, fine_num); \
    glue_type __attribute__((register)) sum[fine_num]; \
    /*glue_type __attribute__((register)) bias_conv_internal[filters]; \
    glue_type __attribute__((register)) Num_conv_internal[filters]; \
    glue_type __attribute__((register)) N_conv_internal[filters]; \
    glue_type __attribute__((register)) bias_temp[fine_num]; \
    glue_type __attribute__((register)) Num_temp[fine_num]; \
    glue_type __attribute__((register)) N_temp[fine_num];*/ \
    /*for (uint filter_index = 0; filter_index < filters - fine_num; filter_index += fine_num ) { \
        _Pragma("unroll")      \
        for(uint fine_index=0 ; fine_index < fine_num ; ++fine_index){ \
            bias_conv_internal[filters - fine_num + fine_index] = bias_conv[filter_index + fine_index]; \
            Num_conv_internal[filters - fine_num + fine_index] = Num_conv[filter_index + fine_index]; \
            N_conv_internal[filters - fine_num + fine_index] = N_conv[filter_index + fine_index]; \
        } \
        _Pragma("unroll")      \
        for (uint j = 0; j < filters - fine_num; j++) { \
            bias_conv_internal[j] = bias_conv_internal[j + fine_num]; \
            Num_conv_internal[j] = Num_conv_internal[j + fine_num]; \
            N_conv_internal[j] = N_conv_internal[j + fine_num]; \
        } \
    } \
    _Pragma("unroll")      \
    for(uint fine_index=0 ; fine_index < fine_num ; ++fine_index){ \
        bias_conv_internal[filters - fine_num + fine_index] = bias_conv[filters - fine_num + fine_index]; \
        Num_conv_internal[filters - fine_num + fine_index] = Num_conv[filters - fine_num + fine_index]; \
        N_conv_internal[filters - fine_num + fine_index] = N_conv[filters - fine_num + fine_index]; \
    } */\
    for (uint pixel_index = 0; pixel_index < rows * cols; pixel_index++) { \
        for (uint filter_index = 0; filter_index < filters; filter_index += fine_num) { \
            _Pragma("unroll")      \
            for(uint fine_index=0; fine_index < fine_num; fine_index++) { \
                sum[fine_index] = read_channel_intel(channel_in[fine_index]);\
            } \
            /*_Pragma("unroll")   \
            for(uint fine_index=0 ; fine_index < fine_num ; ++fine_index){ \
                bias_temp[fine_index] = bias_conv_internal[fine_index]; \
                Num_temp[fine_index] = Num_conv_internal[fine_index]; \
                N_temp[fine_index] = N_conv_internal[fine_index]; \
            } \
            _Pragma("unroll")      \
            for (uint j = 0; j < filters - fine_num; j++) { \
                bias_conv_internal[j] = bias_conv_internal[j + fine_num]; \
                Num_conv_internal[j] = Num_conv_internal[j + fine_num]; \
                N_conv_internal[j] = N_conv_internal[j + fine_num]; \
            } \
            _Pragma("unroll")      \
            for(uint fine_index=0 ; fine_index < fine_num ; ++fine_index){ \
                bias_conv_internal[filters - fine_num + fine_index] = bias_temp[fine_index]; \
                Num_conv_internal[filters - fine_num + fine_index] = Num_temp[fine_index]; \
                N_conv_internal[filters - fine_num + fine_index] = N_temp[fine_index]; \
            } */\
            _Pragma("unroll")      \
            for(uint fine_index=0; fine_index < fine_num; fine_index++) { \
                /*printf("%d\n", sum);*/    \
                uchar k = filter_index + fine_index;  \
                glue_type temp = rightShiftWithRound((bias_conv[k] + sum[fine_index]) * Num_conv[k], N_conv[k]); \
                data_type glue_out = mul(temp + zero_point_o, sigmoid(temp), zero_point_o, zero_point_i_sig, zero_point_o_mul, Num_mul, N_mul); \
                write_channel_intel(channel_out[fine_index], glue_out);  \
            } \
        } \
    } \

/**
    @brief 对所有输出结果进行累加整合，并进行激活和量化
    @param layer kernel函数名称
    @param channel_in 二维输入通道数组
    @param channel_out 二维输出通道数组
    @param coarse_num 粗粒度并行个数
    @param fine_num 细粒度并行个数
    @param rows 输入图片的高像素个数
    @param cols 输入图片的宽像素个数
    @param filters 输出图片的总通道个数（过滤器数量）
    @param glue_type 累加结果的元素类型
 */
#define GEN_GLUE(layer, channel_in, channel_out, coarse_num, fine_num, sigmoid, bias_conv, Num_conv, N_conv, zero_point_o, zero_point_i_sig, zero_point_o_mul, Num_mul, N_mul, rows, cols, filters, data_type, glue_type) \
__kernel        \
__attribute__((max_global_work_dim(0)))     \
__attribute__((autorun))           \
void glue_##layer() {     \
    __GLUE(channel_in, channel_out, coarse_num, fine_num, sigmoid, bias_conv, Num_conv, N_conv, zero_point_o, zero_point_i_sig, zero_point_o_mul, Num_mul, N_mul, rows, cols, filters, data_type, glue_type) \
}

#endif