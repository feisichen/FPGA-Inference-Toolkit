#ifndef MEM_READ_H_
#define MEM_READ_H_

/**
    @brief 将host提供的源地址中的数据写入channel中
    @note 图片通道交织地送给channel
    @param name 核函数命名用
    @param channel_out 一维输出通道数组
    @param coarse_num 粗粒度并行个数
    @param rows 图片高像素个数
    @param cols 图片宽像素个数
    @param channels 总图片通道个数
 */
#define MEM_READ(coarse, name, channel_out, size)     \
__kernel    \
__attribute__((max_global_work_dim(0)))    \
void mem_read_##name(__global char *restrict image) {      \
    uint index = 0; \
    for (uint i = 0; i < size / coarse; ++i) {     \
        _Pragma("unroll")   \
        for (uint j = 0; j < coarse; ++j) {     \
            write_channel_intel(channel_out[j], image[index++]);      \
        } \
    }      \
}

#endif