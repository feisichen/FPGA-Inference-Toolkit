#ifndef ACCUM_H_
#define ACCUM_H_

#define __ACCUM(channel_in, channel_out, coarse_num, fine_num, rows, cols, channels, filters, accum_type) \
    const uint channels_per_group = DIVIDE(channels, coarse_num); \
    const uint filters_per_group = DIVIDE(filters, fine_num); \
    accum_type __attribute__((register)) cache[filters]; \
    accum_type __attribute__((register)) res[fine_num]; \
                                                \
    for (uint i = 0; i < rows * cols; i++) { \
        _Pragma("unroll")      \
        for (uint j = 0; j < filters_per_group; j++) { \
            cache[j] = 0; \
        } \
        for (uint k = 0; k < channels_per_group * filters; k += fine_num) { \
            _Pragma("unroll")      \
            for(uint fine_index=0 ; fine_index < fine_num ; ++fine_index){ \
                res[fine_index] = cache[fine_index] + read_channel_intel(channel_in[fine_index]);  \
            } \
            _Pragma("unroll")   \
            for (uint j = 0; j < filters - fine_num; j++) { \
                cache[j] = cache[j + fine_num];    \
            } \
            if(k < channels_per_group * filters - filters){ \
                _Pragma("unroll")      \
                for(uint fine_index=0 ; fine_index < fine_num ; ++fine_index){ \
                    cache[filters - fine_num + fine_index] = res[fine_index]; \
                } \
            } else { \
                _Pragma("unroll")      \
                for(uint fine_index=0 ; fine_index < fine_num ; ++fine_index){ \
                    write_channel_intel(channel_out[fine_index], res[fine_index]); \
                } \
            } \
        } \
    } 


// #define __ACCUM(channel_in, channel_out, coarse_num, fine_num, rows, cols, channels, filters, accum_type) \
//     const uint channels_per_group = DIVIDE(channels, coarse_num); \
//     const uint filters_per_group = DIVIDE(filters, fine_num); \
//     accum_type cache[filters_per_group]; \
//     const int coarse_index = get_compute_id(0);  \
//     const int fine_index = get_compute_id(1);  \
//                                                 \
//     for (uint i = 0; i < rows * cols; i++) { \
//         _Pragma("unroll")      \
//         for (uint filter_index = 0; filter_index < filters_per_group; filter_index++) { \
//             cache[filter_index] = 0; \
//         } \
//         for (uint channel_index = 0; channel_index < channels_per_group; channel_index++) { \
//             for (uint filter_index = 0; filter_index < filters_per_group; filter_index++) { \
//                 cache[filter_index] += read_channel_intel(channel_in[coarse_index][fine_index]); \
//             } \
//         } \
//         for (uint filter_index = 0; filter_index < filters_per_group; filter_index++){ \
//             write_channel_intel(channel_out[coarse_index][fine_index], cache[filter_index]); \
//         } \
//     } 

/**
    @brief 累加相应输出（图片）通道的卷积结果
    @param layer kernel函数名称
    @param channel_in 二维输入通道数组
    @param channel_out 二维输出通道数组
    @param coarse_num 粗粒度并行个数
    @param fine_num 细粒度并行个数
    @param rows 输入图片的高像素个数
    @param cols 输入图片的宽像素个数
    @param channels 输入图片的总通道个数
    @param filters 输出图片的总通道个数（过滤器数量）
    @param accum_type 累加结果的元素类型
 */
#define GEN_ACCUM(layer, channel_in, channel_out, coarse_num, fine_num, rows, cols, channels, filters, accum_type) \
__kernel        \
__attribute__((max_global_work_dim(0)))     \
__attribute__((autorun))    \
void accum_##layer() {     \
    __ACCUM(channel_in, channel_out, coarse_num, fine_num, rows, cols, channels, filters, accum_type) \
}

#endif