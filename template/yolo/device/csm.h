#ifndef CSM_H_
#define CSM_H_
#include "layer_info.h"
#include "type.h"
#include "sliding_window.h"
//#include "fork.h"
#include "conv.h"
#include "accum.h"
#include "glue.h"
#include "sigmoid.h"

#define GEN_CSM_LAYER(input_width, output_width, layer, channel_in, channel_out, coarse, datatype, acc_datatype) \
/*channel KERNEL_TYPE(kernel_size, kernel_size, datatype) sw_out_##layer __attribute__((depth(0)));     \
channel datatype sw_fork_com_##layer __attribute__((depth(0)));*/ \
channel datatype sw_out_##layer[KERNEL_SIZE_LAYER_##layer * KERNEL_SIZE_LAYER_##layer][input_width] __attribute__((depth(1))); \
/*channel datatype sw_conv_com_##layer __attribute__((depth(WINDOW_CHANNEL_DEPTH)));*/ \
GEN_SLIDING_WINDOW(input_width, layer, channel_in, sw_out_##layer, ZERO_POINT_I_LAYER_##layer, INPUT_SIZE_LAYER_##layer, INPUT_SIZE_LAYER_##layer, INPUT_CHANNELS_LAYER_##layer, PAD_LAYER_##layer, PAD_LAYER_##layer, PAD_LAYER_##layer, PAD_LAYER_##layer, STRIDE_LAYER_##layer, STRIDE_LAYER_##layer, KERNEL_SIZE_LAYER_##layer, KERNEL_SIZE_LAYER_##layer, datatype) \
/*GEN_FORK(layer, sw_out_##layer, fork_out_##layer, fork_conv_com_##layer, sw_fork_com_##layer, OUTPUT_SIZE_LAYER_##layer, OUTPUT_SIZE_LAYER_##layer, INPUT_CHANNELS_LAYER_##layer, KERNEL_SIZE_LAYER_##layer, KERNEL_SIZE_LAYER_##layer, datatype)*/ \
GEN_CONV(input_width, output_width, coarse, layer, sw_out_##layer, channel_out, ZERO_POINT_I_LAYER_##layer, sigmoid_##layer, bias_conv_##layer, num_conv_##layer, n_conv_##layer, ZERO_POINT_O_LAYER_##layer, ZERO_POINT_I_SIG_LAYER_##layer, ZERO_POINT_O_MUL_LAYER_##layer, NUM_MUL_LAYER_##layer, N_MUL_LAYER_##layer, weights_##layer, OUTPUT_SIZE_LAYER_##layer, OUTPUT_SIZE_LAYER_##layer, INPUT_CHANNELS_LAYER_##layer, OUTPUT_CHANNELS_LAYER_##layer, KERNEL_SIZE_LAYER_##layer, KERNEL_SIZE_LAYER_##layer, NUM_BANKS_LAYER_##layer, datatype, datatype, acc_datatype)

#define GEN_PIPE_CSM_LAYER(input_width, output_width, layer, channel_in, channel_out, coarse, datatype, acc_datatype) \
/*channel KERNEL_TYPE(kernel_size, kernel_size, datatype) sw_out_##layer __attribute__((depth(0)));     \
channel datatype sw_fork_com_##layer __attribute__((depth(0)));*/ \
channel datatype sw_out_##layer[KERNEL_SIZE_LAYER_##layer * KERNEL_SIZE_LAYER_##layer + 1][input_width] __attribute__((depth(1))); \
/*channel datatype sw_conv_com_##layer __attribute__((depth(WINDOW_CHANNEL_DEPTH)));*/ \
GEN_SLIDING_WINDOW_PIPE(input_width, layer, channel_in, sw_out_##layer, ZERO_POINT_I_LAYER_##layer, INPUT_SIZE_LAYER_##layer, INPUT_SIZE_LAYER_##layer, INPUT_CHANNELS_LAYER_##layer, PAD_LAYER_##layer, PAD_LAYER_##layer, PAD_LAYER_##layer, PAD_LAYER_##layer, STRIDE_LAYER_##layer, STRIDE_LAYER_##layer, KERNEL_SIZE_LAYER_##layer, KERNEL_SIZE_LAYER_##layer, datatype) \
/*GEN_FORK(layer, sw_out_##layer, fork_out_##layer, fork_conv_com_##layer, sw_fork_com_##layer, OUTPUT_SIZE_LAYER_##layer, OUTPUT_SIZE_LAYER_##layer, INPUT_CHANNELS_LAYER_##layer, KERNEL_SIZE_LAYER_##layer, KERNEL_SIZE_LAYER_##layer, datatype)*/ \
GEN_PIPE_CONV(input_width, output_width, coarse, layer, sw_out_##layer, channel_out, ZERO_POINT_I_LAYER_##layer, sigmoid_##layer, bias_conv_##layer, num_conv_##layer, n_conv_##layer, ZERO_POINT_O_LAYER_##layer, ZERO_POINT_I_SIG_LAYER_##layer, ZERO_POINT_O_MUL_LAYER_##layer, NUM_MUL_LAYER_##layer, N_MUL_LAYER_##layer, weights_##layer, OUTPUT_SIZE_LAYER_##layer, OUTPUT_SIZE_LAYER_##layer, INPUT_CHANNELS_LAYER_##layer, OUTPUT_CHANNELS_LAYER_##layer, KERNEL_SIZE_LAYER_##layer, KERNEL_SIZE_LAYER_##layer, NUM_BANKS_LAYER_##layer, datatype, datatype, acc_datatype)

#define GEN_FINE_CSM_LAYER(input_width, output_width, layer, channel_in, channel_out, coarse, datatype, acc_datatype) \
/*channel KERNEL_TYPE(kernel_size, kernel_size, datatype) sw_out_##layer __attribute__((depth(0)));     \
channel datatype sw_fork_com_##layer __attribute__((depth(0)));*/ \
channel datatype sw_out_##layer[KERNEL_SIZE_LAYER_##layer * KERNEL_SIZE_LAYER_##layer][input_width] __attribute__((depth(1))); \
/*channel datatype sw_conv_com_##layer __attribute__((depth(WINDOW_CHANNEL_DEPTH)));*/ \
GEN_SLIDING_WINDOW(input_width, layer, channel_in, sw_out_##layer, ZERO_POINT_I_LAYER_##layer, INPUT_SIZE_LAYER_##layer, INPUT_SIZE_LAYER_##layer, INPUT_CHANNELS_LAYER_##layer, PAD_LAYER_##layer, PAD_LAYER_##layer, PAD_LAYER_##layer, PAD_LAYER_##layer, STRIDE_LAYER_##layer, STRIDE_LAYER_##layer, KERNEL_SIZE_LAYER_##layer, KERNEL_SIZE_LAYER_##layer, datatype) \
/*GEN_FORK(layer, sw_out_##layer, fork_out_##layer, fork_conv_com_##layer, sw_fork_com_##layer, OUTPUT_SIZE_LAYER_##layer, OUTPUT_SIZE_LAYER_##layer, INPUT_CHANNELS_LAYER_##layer, KERNEL_SIZE_LAYER_##layer, KERNEL_SIZE_LAYER_##layer, datatype)*/ \
GEN_FINE_CONV(input_width, output_width, coarse, layer, sw_out_##layer, channel_out, ZERO_POINT_I_LAYER_##layer, sigmoid_##layer, bias_conv_##layer, num_conv_##layer, n_conv_##layer, ZERO_POINT_O_LAYER_##layer, ZERO_POINT_I_SIG_LAYER_##layer, ZERO_POINT_O_MUL_LAYER_##layer, NUM_MUL_LAYER_##layer, N_MUL_LAYER_##layer, weights_##layer, OUTPUT_SIZE_LAYER_##layer, OUTPUT_SIZE_LAYER_##layer, INPUT_CHANNELS_LAYER_##layer, OUTPUT_CHANNELS_LAYER_##layer, KERNEL_SIZE_LAYER_##layer, KERNEL_SIZE_LAYER_##layer, NUM_BANKS_LAYER_##layer, datatype, datatype, acc_datatype)

#endif