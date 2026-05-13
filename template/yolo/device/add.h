#ifndef ADD_H_
#define ADD_H_
#include "add_info.h"


#define __ADD_CACHE(channel_in,  channel_out, num_filters, height, width, Num_add_x, N_add_x, Num_add_y, N_add_y, zero_point_ix_add, zero_point_iy_add, zero_point_o_add, datatype) { \
    datatype x, y, sum;      \
    const uint buffer_size = height * width * num_filters; \
    char __attribute__((register)) buffer[buffer_size]; \
    for(uint in_index = 0; in_index < 2; in_index++){ \
        for (uint i = 0; i < height; ++i) {     \
            for (uint j = 0; j < width; ++j) {      \
                for (uint k = 0; k < num_filters; ++k) {    \
                    if(in_index == 0){ \
                        y = read_channel_intel(channel_in[0]) - zero_point_ix_add;   \
                        _Pragma("unroll")                   \
                        for (uint buffer_idx = 0; buffer_idx < buffer_size - 1; ++buffer_idx) {    \
                            buffer[buffer_idx] = buffer[buffer_idx + 1];          \
                        }       \
                        buffer[buffer_size - 1] = y;        \
                    } else { \
                        x = read_channel_intel(channel_in[1]) - zero_point_iy_add;   \
                        y = buffer[0]; \
                        _Pragma("unroll")                   \
                        for (uint buffer_idx = 0; buffer_idx < buffer_size - 1; ++buffer_idx) {    \
                            buffer[buffer_idx] = buffer[buffer_idx + 1];          \
                        }       \
                        sum = x * Num_add_x + y * Num_add_y;        \
                        sum = (sum >> N_add_x) + ((sum >> (N_add_x - 1)) & 1) + zero_point_o_add;   \
                        write_channel_intel(channel_out, sum < -128 ? -128 : sum);  \
                    } \
                }   \
            }   \
        }   \
    } \
}

#define GEN_ADD_CACHE(layer, layer_in, layer_out, datatype)    \
__kernel        \
__attribute__((max_global_work_dim(0)))     \
__attribute__((autorun))    \
void add_##layer() {     \
    __ADD_CACHE(layer_in, layer_out, FILTERS_ADD_##layer, HEIGHT_ADD_##layer, WIDTH_ADD_##layer, NUM_ADD_X_##layer, N_ADD_X_##layer, NUM_ADD_Y_##layer, N_ADD_Y_##layer, ZERO_POINT_IX_ADD_##layer, ZERO_POINT_IY_ADD_##layer, ZERO_POINT_O_ADD_##layer, datatype) \
}

#define __ADD_NONGLOBAL(input_width, channel_in_x, channel_in_y, channel_out, num_filters, height, width, Num_add_x, N_add_x, Num_add_y, N_add_y, zero_point_ix_add, zero_point_iy_add, zero_point_o_add, datatype) { \
    datatype x, y, sum;      \
    for (uint i = 0; i < height; ++i) {     \
        for (uint j = 0; j < width; ++j) {      \
            for (uint k = 0; k < num_filters; k += input_width) {    \
                _Pragma("unroll")   \
                for (uint input_idx = 0; input_idx < input_width; ++input_idx) {        \
                    x = read_channel_intel(channel_in_x[input_idx]) - zero_point_ix_add;   \
                    y = read_channel_intel(channel_in_y[input_idx]) - zero_point_iy_add;   \
                    sum = x * Num_add_x + y * Num_add_y;        \
                    sum = (sum >> N_add_x) + ((sum >> (N_add_x - 1)) & 1) + zero_point_o_add;   \
                    write_channel_intel(channel_out[input_idx], sum < -128 ? -128 : sum);  \
                } \
            }   \
        }   \
    }   \
}

#define GEN_ADD_NONGLOBAL(input_width, layer, layer_in_x, layer_in_y, layer_out, datatype)    \
__kernel        \
__attribute__((max_global_work_dim(0)))     \
__attribute__((autorun))    \
void add_##layer() {     \
    __ADD_NONGLOBAL(input_width, layer_in_x, layer_in_y, layer_out, FILTERS_ADD_##layer, HEIGHT_ADD_##layer, WIDTH_ADD_##layer, NUM_ADD_X_##layer, N_ADD_X_##layer, NUM_ADD_Y_##layer, N_ADD_Y_##layer, ZERO_POINT_IX_ADD_##layer, ZERO_POINT_IY_ADD_##layer, ZERO_POINT_O_ADD_##layer, datatype) \
}

#define __ADD_GLOBAL(global_in_x, channel_in_y, channel_out, num_filters, height, width, Num_add_x, N_add_x, Num_add_y, N_add_y, zero_point_ix_add, zero_point_iy_add, zero_point_o_add, datatype) { \
    datatype x, y, sum;      \
    uint x_index = 0; \
    for (uint i = 0; i < height; ++i) {     \
        for (uint j = 0; j < width; ++j) {      \
            for (uint k = 0; k < num_filters; ++k) {    \
                x = global_in_x[x_index++] - zero_point_ix_add;   \
                y = read_channel_intel(channel_in_y) - zero_point_iy_add;   \
                sum = x * Num_add_x + y * Num_add_y;        \
                sum = (sum >> N_add_x) + ((sum >> (N_add_x - 1)) & 1) + zero_point_o_add;   \
                write_channel_intel(channel_out, sum < -128 ? -128 : sum);  \
            }   \
        }   \
    }   \
}

#define GEN_ADD_GLOBAL(layer, layer_in_y, layer_out, datatype)    \
__kernel        \
__attribute__((max_global_work_dim(0)))     \
void add_##layer(__global datatype *restrict input_x) {     \
    __ADD_GLOBAL(input_x, layer_in_y, layer_out, FILTERS_ADD_##layer, HEIGHT_ADD_##layer, WIDTH_ADD_##layer, NUM_ADD_X_##layer, N_ADD_X_##layer, NUM_ADD_Y_##layer, N_ADD_Y_##layer, ZERO_POINT_IX_ADD_##layer, ZERO_POINT_IY_ADD_##layer, ZERO_POINT_O_ADD_##layer, datatype) \
}
#endif