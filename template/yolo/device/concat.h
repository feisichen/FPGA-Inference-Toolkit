#ifndef CONCAT_H_
#define CONCAT_H_
#include "concat_info.h"

#define __CONCAT_QUANT_NONGLOBAL(input_width_1, input_width_2, channel_in_x, channel_in_y, channel_out, num_filters_x, num_filters_y, height, width, Num_concat_x, N_concat_x, Num_concat_y, N_concat_y, zero_point_ix_concat, zero_point_iy_concat, zero_point_o_concat, datatype) {  \
    int tmp, sum;  \
    datatype pixel;    \
                \
    for (uint i = 0; i < height; ++i) {  \
        for (uint j = 0; j < width; ++j) {      \
            for (uint k = 0; k < num_filters_y; k += input_width_2){            \
                _Pragma("unroll")   \
                for (uint input_idx = 0; input_idx < input_width_2; ++input_idx) {        \
                    pixel = read_channel_intel(channel_in_y[input_idx]);        \
                    tmp = (pixel - zero_point_iy_concat) * Num_concat_y;       \
                    sum = (tmp >> N_concat_y) + ((tmp >> (N_concat_y - 1)) & 1) + zero_point_o_concat;  \
                    write_channel_intel(channel_out[input_idx], sum < -128 ? -128 : sum);      \
                } \
            } \
            for (uint k = 0; k < num_filters_x; k += input_width_1){           \
                _Pragma("unroll")   \
                for (uint input_idx = 0; input_idx < input_width_1; ++input_idx) {        \
                    pixel = read_channel_intel(channel_in_x[input_idx]);        \
                    tmp = (pixel - zero_point_ix_concat) * Num_concat_x;       \
                    sum = (tmp >> N_concat_x) + ((tmp >> (N_concat_x - 1)) & 1) + zero_point_o_concat;      \
                    write_channel_intel(channel_out[input_idx], sum < -128 ? -128 : sum);      \
                } \
            }     \
        }    \
    }   \
}

#define GEN_CONCAT_QUANT_NONGLOBAL(input_width_1, input_width_2, layer, layer_in_x, layer_in_y, layer_out, datatype)   \
__kernel        \
__attribute__((max_global_work_dim(0)))     \
__attribute__((autorun))       \
void concat_quant_##layer() {     \
    __CONCAT_QUANT_NONGLOBAL(input_width_1, input_width_2, layer_in_x, layer_in_y, layer_out, FILTERS_X_CONCAT_##layer, FILTERS_Y_CONCAT_##layer, HEIGHT_CONCAT_##layer, WIDTH_CONCAT_##layer, NUM_CONCAT_X_##layer, N_CONCAT_X_##layer, NUM_CONCAT_Y_##layer, N_CONCAT_Y_##layer, ZERO_POINT_IX_CONCAT_##layer, ZERO_POINT_IY_CONCAT_##layer, ZERO_POINT_O_CONCAT_##layer, datatype) \
}

#define __CONCAT_QUANT_GLOBAL(global_in_x, channel_in_y, channel_out, num_filters_x, num_filters_y, height, width, Num_concat_x, N_concat_x, Num_concat_y, N_concat_y, zero_point_ix_concat, zero_point_iy_concat, zero_point_o_concat, datatype) {  \
    int tmp, sum;  \
    datatype pixel;    \
    uint x_index = 0; \
    for (uint i = 0;i < height; ++i) {  \
        for (uint j = 0; j < width; ++j) {      \
            for (uint k = 0; k < num_filters_x + num_filters_y; ++k){       \
                if (k < num_filters_y) {                \
                    pixel = read_channel_intel(channel_in_y);        \
                    tmp = (pixel - zero_point_iy_concat) * Num_concat_y;       \
                    sum = (tmp >> N_concat_y) + ((tmp >> (N_concat_y - 1)) & 1) + zero_point_o_concat;  \
                    write_channel_intel(channel_out, sum < -128 ? -128 : sum);  \
                } else {        \
                    pixel = global_in_x[x_index++];        \
                    tmp = (pixel - zero_point_ix_concat) * Num_concat_x;       \
                    sum = (tmp >> N_concat_x) + ((tmp >> (N_concat_x - 1)) & 1) + zero_point_o_concat;      \
                    write_channel_intel(channel_out, sum < -128 ? -128 : sum);      \
                }     \
            }     \
        }    \
    }   \
}

#define GEN_CONCAT_QUANT_GLOBAL(layer, layer_in_y, layer_out, num_filters_x, num_filters_y, height, width, datatype)   \
__kernel        \
__attribute__((max_global_work_dim(0)))     \
void concat_quant_##layer(__global datatype    *restrict input_x) {     \
    __CONCAT_QUANT_GLOBAL(input_x, sw_in_##layer_in_y, sw_in_##layer_out, num_filters_x, num_filters_y, height, width, concat_info_##layer.Num_concat_x, concat_info_##layer.N_concat_x, concat_info_##layer.Num_concat_y, concat_info_##layer.N_concat_y, concat_info_##layer.zero_point_ix_concat, concat_info_##layer.zero_point_iy_concat, concat_info_##layer.zero_point_o_concat, datatype) \
}

#define __CONCAT(channel_in_x, channel_in_y, channel_out, num_filters_x, num_filters_y, height, width, datatype) {  \
    datatype pixel;    \
                    \
    for (uint i = 0;i < height; ++i) {  \
        for (uint j = 0; j < width; ++j) {      \
            for (uint k = 0; k < num_filters_x + num_filters_y; ++k){       \
                if (k < num_filters_y) {                \
                    pixel = read_channel_intel(channel_in_y);        \
                } else {        \
                    pixel = read_channel_intel(channel_in_x);        \
                }     \
                write_channel_intel(channel_out, pixel);  \
            }     \
        }    \
    }   \
}

#define GEN_CONCAT(layer, layer_in_x, layer_in_y, layer_out, datatype)   \
__kernel        \
__attribute__((max_global_work_dim(0)))     \
__attribute__((autorun))    \
void concat_##layer() {     \
    __CONCAT(layer_in_x, layer_in_y, layer_out, FILTERS_X_CONCAT_##layer, FILTERS_Y_CONCAT_##layer, HEIGHT_CONCAT_##layer, WIDTH_CONCAT_##layer, datatype) \
}

#endif
