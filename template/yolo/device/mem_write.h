#ifndef MEM_WRITE_H_
#define MEM_WRITE_H_

/**
    @brief 从channel读取数据至全局内存空间中
    @param name 核函数命名
    @param channel_in 一维输入通道数组
    @param fine 输入通道个数 
    @param size 写入全局内存的总字节数
 */
#define MEM_WRITE(coarse, name, channel_in, size)       \
__kernel    \
__attribute__((max_global_work_dim(0)))      \
void mem_write_##name(__global char *restrict data) {      \
    uint index = 0; \
    for (uint i = 0; i < size / coarse; ++i) {      \
        _Pragma("unroll")   \
        for (uint j = 0; j < coarse; ++j) {     \
            data[index++] = read_channel_intel(channel_in[j]);   \
        /*printf("[%d]:%d\n", i + j, data[i + j]);*/   \
        } \
    }      \
}

#endif