#ifndef CONNECT_H_
#define CONNECT_H_

#define GEN_MERGE(name, channel_in, channel_out, fine_prev, rows, cols, channels, datatype) \
__kernel        \
__attribute__((max_global_work_dim(0)))     \
__attribute__((autorun))    \
void merge_##name() {     \
    for (uint i = 0; i < rows * cols * channels; i += fine_prev) {  \
        for (uchar j = 0; j < fine_prev; ++j) { \
            datatype pixel = read_channel_intel(channel_in[j]); \
            write_channel_intel(channel_out, pixel);    \
        }   \
    }   \
}

/**
    @brief 将图片数据交织地拆分给多个channel
    @param name kernel函数名称
    @param channel_in 输入channel
    @param channel_out 输出channel
    @param coarse_next 输出channel个数
    @param rows 图片高像素个数
    @param cols 图片宽像素个数
    @param channels 图片总通道个数
    @param datatype 图片像素点数据类型
 */
#define GEN_SPLIT(name, channel_in, channel_out, coarse_next, rows, cols, channels, datatype)  \
__kernel        \
__attribute__((max_global_work_dim(0)))     \
__attribute__((autorun))    \
void splt_##name() {   \
    for (uint i = 0; i < rows * cols * channels; i += channels) {     \
        _Pragma("unroll")   \
        for (uchar j = 0; j < channels; j += coarse_next) {      \
            _Pragma("unroll")       \
            for (uchar k = 0; k < coarse_next; ++k) {        \
                datatype pixel = read_channel_intel(channel_in);        \
                write_channel_intel(channel_out[k], pixel);      \
            }     \
        }     \
    }      \
}

/**
    @brief 用于层间channel数不同时的衔接
    @param name kernel函数名称
    @param channel_in 输入channel
    @param channel_out 输出channel
    @param fine_prev 输入channel个数
    @param coarse_next 输出channel个数
    @param rows 图片高像素个数
    @param cols 图片宽像素个数
    @param channels 图片总通道个数
    @param datatype 图片像素点数据类型
 */
#define GEN_CONNECT(name, channel_in, channel_out, fine_prev, coarse_next, rows, cols, channels, datatype)   \
channel datatype merge_out_##name __attribute__((depth(0)));    \
GEN_MERGE(name, channel_in, merge_out_##name, fine_prev, rows, cols, channels, datatype)    \
GEN_SPLIT(name, merge_out_##name, channel_out, coarse_next, rows, cols, channels, datatype)

#endif