#ifndef YOLOV5N_CONFIGS_H
#define YOLOV5N_CONFIGS_H
#include "Layer.h"
#include "weights.h"
#include "Functions.h"
#include "layer_info.h"
#include "add_info.h"
#include "mul_info.h"
#include "concat_info.h"
#include "maxpool_info.h"
#include "sigmoid.h"
#include "Mem.h"
#include "Box.h"

#define REGISTET_CSM(layer, type, accum_type) \
csm<type \
, accum_type \
, INPUT_SIZE_LAYER_##layer \
, INPUT_SIZE_LAYER_##layer \
, INPUT_CHANNELS_LAYER_##layer \
, OUTPUT_CHANNELS_LAYER_##layer \
, KERNEL_SIZE_LAYER_##layer \
, KERNEL_SIZE_LAYER_##layer \
, PAD_LAYER_##layer \
, PAD_LAYER_##layer \
, PAD_LAYER_##layer \
, PAD_LAYER_##layer \
, STRIDE_LAYER_##layer \
, STRIDE_LAYER_##layer \
, weights_##layer \
, sigmoid_##layer \
, bias_conv_##layer \
, num_conv_##layer \
, n_conv_##layer \
, ZERO_POINT_I_LAYER_##layer \
, ZERO_POINT_O_LAYER_##layer \
, ZERO_POINT_I_SIG_LAYER_##layer \
, ZERO_POINT_O_MUL_LAYER_##layer \
, NUM_MUL_LAYER_##layer \
, N_MUL_LAYER_##layer \
>

#define REGISTET_CSM_LAST(layer, type, accum_type) \
csm_last<type \
, accum_type \
, INPUT_SIZE_LAYER_##layer \
, INPUT_SIZE_LAYER_##layer \
, INPUT_CHANNELS_LAYER_##layer \
, OUTPUT_CHANNELS_LAYER_##layer \
, KERNEL_SIZE_LAYER_##layer \
, KERNEL_SIZE_LAYER_##layer \
, PAD_LAYER_##layer \
, PAD_LAYER_##layer \
, PAD_LAYER_##layer \
, PAD_LAYER_##layer \
, STRIDE_LAYER_##layer \
, STRIDE_LAYER_##layer \
, weights_##layer \
, bias_conv_##layer \
, num_conv_##layer \
, n_conv_##layer \
, ZERO_POINT_I_LAYER_##layer \
, ZERO_POINT_O_LAYER_##layer \
>

#define REGISTET_CONCAT(layer, type) \
concat<type \
, FILTERS_X_CONCAT_##layer \
, FILTERS_Y_CONCAT_##layer \
, HEIGHT_CONCAT_##layer \
, WIDTH_CONCAT_##layer \
, NUM_CONCAT_X_##layer \
, N_CONCAT_X_##layer \
, NUM_CONCAT_Y_##layer \
, N_CONCAT_Y_##layer \
, ZERO_POINT_IX_CONCAT_##layer \
, ZERO_POINT_IY_CONCAT_##layer \
, ZERO_POINT_O_CONCAT_##layer \
>

#define REGISTET_CONCAT_WITHOUT_QUANT(layer, type) \
concat_without_quant<type \
, FILTERS_X_CONCAT_##layer \
, FILTERS_Y_CONCAT_##layer \
, HEIGHT_CONCAT_##layer \
, WIDTH_CONCAT_##layer \
>

#define REGISTET_ADD(layer, type) \
add<type \
, FILTERS_ADD_##layer \
, HEIGHT_ADD_##layer \
, WIDTH_ADD_##layer \
, NUM_ADD_X_##layer \
, N_ADD_X_##layer \
, NUM_ADD_Y_##layer \
, N_ADD_Y_##layer \
, ZERO_POINT_IX_ADD_##layer \
, ZERO_POINT_IY_ADD_##layer \
, ZERO_POINT_O_ADD_##layer \
>

#define REGISTET_ADD_WITH_XY(layer, type, name) \
add<type \
, FILTERS_ADD_##name##_##layer \
, HEIGHT_ADD_##name##_##layer \
, WIDTH_ADD_##name##_##layer \
, NUM_ADD_X_##name##_##layer \
, N_ADD_X_##name##_##layer \
, NUM_ADD_Y_##name##_##layer \
, N_ADD_Y_##name##_##layer \
, ZERO_POINT_IX_ADD_##name##_##layer \
, ZERO_POINT_IY_ADD_##name##_##layer \
, ZERO_POINT_O_ADD_##name##_##layer \
>

#define REGISTET_MUL(layer, type, name) \
mul<type \
, FILTERS_MUL_##name##_##layer \
, HEIGHT_MUL_##name##_##layer \
, WIDTH_MUL_##name##_##layer \
, NUM_MUL_##name##_##layer \
, N_MUL_##name##_##layer \
, ZERO_POINT_IX_MUL_##name##_##layer \
, ZERO_POINT_IY_MUL_##name##_##layer \
, ZERO_POINT_O_MUL_##name##_##layer \
>

#define REGISTET_MUL_CONST(layer, type, value, name) \
mul_const<type \
, value \
, FILTERS_MUL_##name##_##layer \
, HEIGHT_MUL_##name##_##layer \
, WIDTH_MUL_##name##_##layer \
, NUM_MUL_##name##_##layer \
, N_MUL_##name##_##layer \
, ZERO_POINT_IX_MUL_##name##_##layer \
, ZERO_POINT_IY_MUL_##name##_##layer \
, ZERO_POINT_O_MUL_##name##_##layer \
>

#define REGISTET_MAXPOOL(layer, type) \
maxpool<type \
, HEIGHT_MAXPOOL_##layer \
, WIDTH_MAXPOOL_##layer \
, CHANNELS_MAXPOOL_##layer \
, KERNEL_SIZE_MAXPOOL_##layer \
, KERNEL_SIZE_MAXPOOL_##layer \
, PAD_MAXPOOL_##layer \
, PAD_MAXPOOL_##layer \
, PAD_MAXPOOL_##layer \
, PAD_MAXPOOL_##layer \
, STRIDE_MAXPOOL_##layer \
, STRIDE_MAXPOOL_##layer \
, ZERO_POINT_MAXPOOL_##layer \
>

#define REGISTET_SPLITX3(layer, type, size1, size2, size3) \
splitx3<type \
, INPUT_SIZE_LAYER_##layer \
, INPUT_SIZE_LAYER_##layer \
, size1 \
, size2 \
, size3 \
>

#define REGISTET_RESIZE(layer, type) \
resize<type \
, OUTPUT_SIZE_LAYER_##layer \
, OUTPUT_SIZE_LAYER_##layer \
, OUTPUT_CHANNELS_LAYER_##layer \
>

#define REGISTET_TRANSPOSE(layer, type, c1) \
transpose<type \
, OUTPUT_SIZE_LAYER_##layer \
, OUTPUT_SIZE_LAYER_##layer \
, OUTPUT_CHANNELS_LAYER_##layer \
, sigmoid_##layer \
, ZERO_POINT_O_LAYER_##layer \
, c1 \
>

template void REGISTET_CSM(0, int8_t, int)(int group_num
        , int group_index
        , std::vector<std::reference_wrapper<std::vector<int8_t, AlignedAllocator<int8_t, 64>>>> inputs
        , std::vector<std::reference_wrapper<std::vector<int8_t, AlignedAllocator<int8_t, 64>>>> outputs
);


#define OUT_SIZE_OF_LAYER(layer) OUTPUT_SIZE_LAYER_##layer * OUTPUT_SIZE_LAYER_##layer * OUTPUT_CHANNELS_LAYER_##layer
#define IN_SIZE_OF_LAYER(layer) INPUT_SIZE_LAYER_##layer * INPUT_SIZE_LAYER_##layer * INPUT_CHANNELS_LAYER_##layer
//模型的描述格式为:{函数名, HOST/DEVICE, 所使用设备的序号(如果是HOST函数则为该函数的并行度), 输入参数列表, 输出参数列表}
//输入/输出参数的描述格式为:{参数名称, 大小} NOTE:(参数名称“input”, “output”为保留关键字, 分别表示该模型的输入输出)

// std::vector<Layer> yolov5n= {{"csm_0", HOST, 20, {{"input", 640*640*3}}, {{"csm_0_out", SIZE_OF_LAYER(0)}}}
//                             ,{"csm_1", HOST, 20, {{"csm_0_out", SIZE_OF_LAYER(0)}}, {{"csm_1_out", SIZE_OF_LAYER(1)}}}
//                             ,{"csm_2", HOST, 20, {{"csm_1_out", SIZE_OF_LAYER(1)}}, {{"csm_2_out", SIZE_OF_LAYER(2)}}}
//                             ,{"csm_3", HOST, 20, {{"csm_2_out", SIZE_OF_LAYER(2)}}, {{"csm_3_out", SIZE_OF_LAYER(3)}}}
//                             ,{"csm_4", HOST, 20, {{"csm_3_out", SIZE_OF_LAYER(3)}}, {{"csm_4_out", SIZE_OF_LAYER(4)}}}
//                             ,{"_add_0", HOST,  1, {{"csm_2_out", SIZE_OF_LAYER(2)}, {"csm_4_out", SIZE_OF_LAYER(4)}}, {{"csm_4_out", SIZE_OF_LAYER(4)}}}
//                             ,{"csm_5", HOST, 20, {{"csm_1_out", SIZE_OF_LAYER(1)}}, {{"csm_5_out", SIZE_OF_LAYER(5)}}}
//                             ,{"concat_0", HOST, 1, {{"csm_4_out", SIZE_OF_LAYER(4)}, {"csm_5_out", SIZE_OF_LAYER(5)}}, {{"concat_0_out", SIZE_OF_LAYER(4) + SIZE_OF_LAYER(5)}}}
//                             ,{"csm_6", HOST, 20, {{"concat_0_out", SIZE_OF_LAYER(4) + SIZE_OF_LAYER(5)}}, {{"csm_6_out", SIZE_OF_LAYER(6)}}}
//                             ,{"csm_7", HOST, 20, {{"csm_6_out", SIZE_OF_LAYER(6)}}, {{"csm_7_out", SIZE_OF_LAYER(7)}}}
//                             ,{"csm_8", HOST, 20, {{"csm_7_out", SIZE_OF_LAYER(7)}}, {{"csm_8_out", SIZE_OF_LAYER(8)}}}
//                             ,{"csm_9", HOST, 20, {{"csm_8_out", SIZE_OF_LAYER(8)}}, {{"csm_9_out", SIZE_OF_LAYER(9)}}}
//                             ,{"csm_10", HOST, 20, {{"csm_9_out", SIZE_OF_LAYER(9)}}, {{"csm_10_out", SIZE_OF_LAYER(10)}}}
//                             ,{"_add_1", HOST,  1, {{"csm_8_out", SIZE_OF_LAYER(8)}, {"csm_10_out", SIZE_OF_LAYER(10)}}, {{"csm_10_out", SIZE_OF_LAYER(10)}}}
//                             ,{"csm_11", HOST, 20, {{"csm_10_out", SIZE_OF_LAYER(10)}}, {{"csm_11_out", SIZE_OF_LAYER(11)}}}
//                             ,{"csm_12", HOST, 20, {{"csm_11_out", SIZE_OF_LAYER(11)}}, {{"csm_12_out", SIZE_OF_LAYER(12)}}}
//                             ,{"_add_2", HOST,  1, {{"csm_10_out", SIZE_OF_LAYER(10)}, {"csm_12_out", SIZE_OF_LAYER(12)}}, {{"csm_12_out", SIZE_OF_LAYER(12)}}}
//                             ,{"csm_13", HOST, 20, {{"csm_7_out", SIZE_OF_LAYER(7)}}, {{"csm_13_out", SIZE_OF_LAYER(13)}}}
//                             ,{"concat_1", HOST, 1, {{"csm_12_out", SIZE_OF_LAYER(12)}, {"csm_13_out", SIZE_OF_LAYER(13)}}, {{"concat_1_out", SIZE_OF_LAYER(12) + SIZE_OF_LAYER(13)}}}
//                             ,{"csm_14", HOST, 20, {{"concat_1_out", SIZE_OF_LAYER(12) + SIZE_OF_LAYER(13)}}, {{"csm_14_out", SIZE_OF_LAYER(14)}}}
//                             ,{"csm_15", HOST, 20, {{"csm_14_out", SIZE_OF_LAYER(14)}}, {{"csm_15_out", SIZE_OF_LAYER(15)}}}
//                             ,{"csm_16", HOST, 20, {{"csm_15_out", SIZE_OF_LAYER(15)}}, {{"csm_16_out", SIZE_OF_LAYER(16)}}}
//                             ,{"csm_17", HOST, 20, {{"csm_16_out", SIZE_OF_LAYER(16)}}, {{"csm_17_out", SIZE_OF_LAYER(17)}}}
//                             ,{"csm_18", HOST, 20, {{"csm_17_out", SIZE_OF_LAYER(17)}}, {{"csm_18_out", SIZE_OF_LAYER(18)}}}
//                             ,{"_add_3", HOST,  1, {{"csm_16_out", SIZE_OF_LAYER(16)}, {"csm_18_out", SIZE_OF_LAYER(18)}}, {{"csm_18_out", SIZE_OF_LAYER(18)}}}
//                             ,{"csm_19", HOST, 20, {{"csm_18_out", SIZE_OF_LAYER(18)}}, {{"csm_19_out", SIZE_OF_LAYER(19)}}}
//                             ,{"csm_20", HOST, 20, {{"csm_19_out", SIZE_OF_LAYER(19)}}, {{"csm_20_out", SIZE_OF_LAYER(20)}}}
//                             ,{"_add_4", HOST,  1, {{"csm_18_out", SIZE_OF_LAYER(18)}, {"csm_20_out", SIZE_OF_LAYER(20)}}, {{"csm_20_out", SIZE_OF_LAYER(20)}}}
//                             ,{"csm_21", HOST, 20, {{"csm_20_out", SIZE_OF_LAYER(20)}}, {{"csm_21_out", SIZE_OF_LAYER(21)}}}
//                             ,{"csm_22", HOST, 20, {{"csm_21_out", SIZE_OF_LAYER(21)}}, {{"csm_22_out", SIZE_OF_LAYER(22)}}}
//                             ,{"_add_5", HOST,  1, {{"csm_20_out", SIZE_OF_LAYER(20)}, {"csm_22_out", SIZE_OF_LAYER(22)}}, {{"csm_22_out", SIZE_OF_LAYER(22)}}}
//                             ,{"csm_23", HOST, 20, {{"csm_15_out", SIZE_OF_LAYER(15)}}, {{"csm_23_out", SIZE_OF_LAYER(23)}}}
//                             ,{"concat_2", HOST, 1, {{"csm_22_out", SIZE_OF_LAYER(22)}, {"csm_23_out", SIZE_OF_LAYER(23)}}, {{"concat_2_out", SIZE_OF_LAYER(22) + SIZE_OF_LAYER(23)}}}
//                             ,{"csm_24", HOST, 20, {{"concat_2_out", SIZE_OF_LAYER(22) + SIZE_OF_LAYER(23)}}, {{"csm_24_out", SIZE_OF_LAYER(24)}}}
//                             ,{"csm_25", HOST, 20, {{"csm_24_out", SIZE_OF_LAYER(24)}}, {{"csm_25_out", SIZE_OF_LAYER(25)}}}
//                             ,{"csm_26", HOST, 20, {{"csm_25_out", SIZE_OF_LAYER(25)}}, {{"csm_26_out", SIZE_OF_LAYER(26)}}}
//                             ,{"csm_27", HOST, 20, {{"csm_26_out", SIZE_OF_LAYER(26)}}, {{"csm_27_out", SIZE_OF_LAYER(27)}}}
//                             ,{"csm_28", HOST, 20, {{"csm_27_out", SIZE_OF_LAYER(27)}}, {{"csm_28_out", SIZE_OF_LAYER(28)}}}
//                             ,{"_add_6", HOST,  1, {{"csm_26_out", SIZE_OF_LAYER(26)}, {"csm_28_out", SIZE_OF_LAYER(28)}}, {{"csm_28_out", SIZE_OF_LAYER(28)}}}
//                             ,{"csm_29", HOST, 20, {{"csm_25_out", SIZE_OF_LAYER(25)}}, {{"csm_29_out", SIZE_OF_LAYER(29)}}}
//                             ,{"concat_3", HOST, 1, {{"csm_28_out", SIZE_OF_LAYER(28)}, {"csm_29_out", SIZE_OF_LAYER(29)}}, {{"concat_3_out", SIZE_OF_LAYER(28) + SIZE_OF_LAYER(29)}}}
//                             ,{"csm_30", HOST, 20, {{"concat_3_out", SIZE_OF_LAYER(28) + SIZE_OF_LAYER(29)}}, {{"csm_30_out", SIZE_OF_LAYER(30)}}}
//                             ,{"csm_31", HOST, 20, {{"csm_30_out", SIZE_OF_LAYER(30)}}, {{"csm_31_out", SIZE_OF_LAYER(31)}}}
//                             ,{"maxpool_0", HOST, 20, {{"csm_31_out", SIZE_OF_LAYER(30)}}, {{"maxpool_0_out", 128 * 20 * 20}}}
//                             ,{"maxpool_1", HOST, 20, {{"maxpool_0_out", 128 * 20 * 20}}, {{"maxpool_1_out", 128 * 20 * 20}}}
//                             ,{"maxpool_2", HOST, 20, {{"maxpool_1_out", 128 * 20 * 20}}, {{"maxpool_2_out", 128 * 20 * 20}}}
//                             ,{"concat_4_1", HOST, 1, {{"csm_31_out", SIZE_OF_LAYER(31)}, {"maxpool_0_out", 128 * 20 * 20}}, {{"concat_4_1_out", SIZE_OF_LAYER(31) + 128 * 20 * 20}}}
//                             ,{"concat_4_2", HOST, 1, {{"concat_4_1_out", SIZE_OF_LAYER(31) + 128 * 20 * 20}, {"maxpool_1_out", 128 * 20 * 20}}, {{"concat_4_2_out", SIZE_OF_LAYER(31) + 128 * 20 * 20 * 2}}}
//                             ,{"concat_4_3", HOST, 1, {{"concat_4_2_out", SIZE_OF_LAYER(31) + 128 * 20 * 20 * 2}, {"maxpool_2_out", 128 * 20 * 20}}, {{"concat_4_3_out", SIZE_OF_LAYER(31) + 128 * 20 * 20 * 3}}}
//                             ,{"csm_32", HOST,  20, {{"concat_4_3_out", 20 * 20 * 512}}, {{"csm_32_out", SIZE_OF_LAYER(32)}}}
//                             ,{"csm_33", HOST,  20, {{"csm_32_out", SIZE_OF_LAYER(32)}}, {{"csm_33_out", SIZE_OF_LAYER(33)}}}
//                             ,{"resize_0", HOST, 1, {{"csm_33_out", SIZE_OF_LAYER(33)}}, {{"resize_0_output", SIZE_OF_LAYER(33) * 4}}}
//                             ,{"concat_5", HOST, 1, {{"resize_0_output", SIZE_OF_LAYER(33) * 4}, {"csm_24_out", SIZE_OF_LAYER(24)}}, {{"concat_5_out", SIZE_OF_LAYER(33) * 4 + SIZE_OF_LAYER(24)}}}
//                             ,{"csm_34", HOST,  20, {{"concat_5_out", SIZE_OF_LAYER(33) + SIZE_OF_LAYER(24)}}, {{"csm_34_out", SIZE_OF_LAYER(34)}}}
//                             ,{"csm_35", HOST,  20, {{"csm_34_out", SIZE_OF_LAYER(34)}}, {{"csm_35_out", SIZE_OF_LAYER(35)}}}
//                             ,{"csm_36", HOST,  20, {{"csm_35_out", SIZE_OF_LAYER(35)}}, {{"csm_36_out", SIZE_OF_LAYER(36)}}}
//                             ,{"csm_37", HOST,  20, {{"concat_5_out", SIZE_OF_LAYER(33) + SIZE_OF_LAYER(24)}}, {{"csm_37_out", SIZE_OF_LAYER(37)}}}
//                             ,{"concat_6", HOST, 1, {{"csm_36_out", SIZE_OF_LAYER(36)}, {"csm_37_out", SIZE_OF_LAYER(37)}}, {{"concat_6_out", SIZE_OF_LAYER(36) + SIZE_OF_LAYER(37)}}}
//                             ,{"csm_38", HOST,  20, {{"concat_6_out", SIZE_OF_LAYER(36) + SIZE_OF_LAYER(37)}}, {{"csm_38_out", SIZE_OF_LAYER(38)}}}
//                             ,{"csm_39", HOST,  20, {{"csm_38_out", SIZE_OF_LAYER(38)}}, {{"csm_39_out", SIZE_OF_LAYER(39)}}}
//                             ,{"resize_1", HOST, 1, {{"csm_39_out", SIZE_OF_LAYER(39)}}, {{"resize_1_output", SIZE_OF_LAYER(39) * 4}}}
//                             ,{"concat_7", HOST, 1, {{"resize_1_output", SIZE_OF_LAYER(39) * 4}, {"csm_14_out", SIZE_OF_LAYER(14)}}, {{"concat_7_out", SIZE_OF_LAYER(39) * 4 + SIZE_OF_LAYER(14)}}}
//                             ,{"csm_40", HOST,  20, {{"concat_7_out", SIZE_OF_LAYER(39) + SIZE_OF_LAYER(14)}}, {{"csm_40_out", SIZE_OF_LAYER(40)}}}
//                             ,{"csm_41", HOST,  20, {{"csm_40_out", SIZE_OF_LAYER(40)}}, {{"csm_41_out", SIZE_OF_LAYER(41)}}}
//                             ,{"csm_42", HOST,  20, {{"csm_41_out", SIZE_OF_LAYER(41)}}, {{"csm_42_out", SIZE_OF_LAYER(42)}}}
//                             ,{"csm_43", HOST,  20, {{"concat_7_out", SIZE_OF_LAYER(39) + SIZE_OF_LAYER(14)}}, {{"csm_43_out", SIZE_OF_LAYER(43)}}}
//                             ,{"concat_8", HOST, 1, {{"csm_42_out", SIZE_OF_LAYER(42)}, {"csm_43_out", SIZE_OF_LAYER(43)}}, {{"concat_8_out", SIZE_OF_LAYER(42) + SIZE_OF_LAYER(43)}}}
//                             ,{"csm_44", HOST,  20, {{"concat_8_out", SIZE_OF_LAYER(42) + SIZE_OF_LAYER(43)}}, {{"csm_44_out", SIZE_OF_LAYER(44)}}}
//                             ,{"csm_45", HOST,  20, {{"csm_44_out", SIZE_OF_LAYER(44)}}, {{"csm_45_out", SIZE_OF_LAYER(45)}}}
//                             ,{"concat_9", HOST, 1, {{"csm_45_out", SIZE_OF_LAYER(45)}, {"csm_39_out", SIZE_OF_LAYER(39)}}, {{"concat_9_out", SIZE_OF_LAYER(45) + SIZE_OF_LAYER(39)}}}
//                             ,{"csm_46", HOST,  20, {{"concat_9_out", SIZE_OF_LAYER(45) + SIZE_OF_LAYER(39)}}, {{"csm_46_out", SIZE_OF_LAYER(46)}}}
//                             ,{"csm_47", HOST,  20, {{"csm_46_out", SIZE_OF_LAYER(46)}}, {{"csm_47_out", SIZE_OF_LAYER(47)}}}
//                             ,{"csm_48", HOST,  20, {{"csm_47_out", SIZE_OF_LAYER(47)}}, {{"csm_48_out", SIZE_OF_LAYER(48)}}}
//                             ,{"csm_49", HOST,  20, {{"concat_9_out", SIZE_OF_LAYER(45) + SIZE_OF_LAYER(39)}}, {{"csm_49_out", SIZE_OF_LAYER(49)}}}
//                             ,{"concat_10", HOST, 1, {{"csm_48_out", SIZE_OF_LAYER(48)}, {"csm_49_out", SIZE_OF_LAYER(49)}}, {{"concat_10_out", SIZE_OF_LAYER(48) + SIZE_OF_LAYER(49)}}}
//                             ,{"csm_50", HOST,  20, {{"concat_10_out", SIZE_OF_LAYER(49) + SIZE_OF_LAYER(49)}}, {{"csm_50_out", SIZE_OF_LAYER(50)}}}
//                             ,{"csm_51", HOST,  20, {{"csm_50_out", SIZE_OF_LAYER(50)}}, {{"csm_51_out", SIZE_OF_LAYER(51)}}}
//                             ,{"concat_11", HOST, 1, {{"csm_51_out", SIZE_OF_LAYER(51)}, {"csm_33_out", SIZE_OF_LAYER(33)}}, {{"concat_11_out", SIZE_OF_LAYER(51) + SIZE_OF_LAYER(33)}}}
//                             ,{"csm_52", HOST,  20, {{"concat_11_out", SIZE_OF_LAYER(51) + SIZE_OF_LAYER(33)}}, {{"csm_52_out", SIZE_OF_LAYER(52)}}}
//                             ,{"csm_53", HOST,  20, {{"csm_52_out", SIZE_OF_LAYER(52)}}, {{"csm_53_out", SIZE_OF_LAYER(53)}}}
//                             ,{"csm_54", HOST,  20, {{"csm_53_out", SIZE_OF_LAYER(53)}}, {{"csm_54_out", SIZE_OF_LAYER(54)}}}
//                             ,{"csm_55", HOST,  20, {{"concat_11_out", SIZE_OF_LAYER(51) + SIZE_OF_LAYER(33)}}, {{"csm_55_out", SIZE_OF_LAYER(55)}}}
//                             ,{"concat_12", HOST, 1, {{"csm_54_out", SIZE_OF_LAYER(54)}, {"csm_55_out", SIZE_OF_LAYER(55)}}, {{"concat_12_out", SIZE_OF_LAYER(54) + SIZE_OF_LAYER(55)}}}
//                             ,{"csm_56", HOST,  20, {{"concat_12_out", SIZE_OF_LAYER(54) + SIZE_OF_LAYER(55)}}, {{"csm_56_out", SIZE_OF_LAYER(56)}}}
//                             ,{"csm_57", HOST,  20, {{"csm_44_out", SIZE_OF_LAYER(44)}}, {{"csm_57_out", SIZE_OF_LAYER(57)}}}
//                             ,{"transpose_0", HOST, 1, {{"csm_57_out", SIZE_OF_LAYER(57)}}, {{"csm_57_out_t", SIZE_OF_LAYER(57)}}}
//                             ,{"splitx3_0", HOST, 1, {{"csm_57_out_t", SIZE_OF_LAYER(57)}}, {{"split_57_x", SIZE_OF_LAYER(57) * 2 / 25}, {"split_57_y", SIZE_OF_LAYER(57) * 2 / 25}, {"split_57_z", SIZE_OF_LAYER(57) * 21 / 25}}}
//                             ,{"mul_0_x", HOST, 1, {{"split_57_x", SIZE_OF_LAYER(57) * 2 / 25}}, {{"split_57_x", SIZE_OF_LAYER(57) * 2 / 25}}}
//                             ,{"add_0", HOST, 1, {{"weights_add_0", SIZE_OF_LAYER(57) * 2 / 25}, {"split_57_x", SIZE_OF_LAYER(57) * 2 / 25}}, {{"split_57_x", SIZE_OF_LAYER(57) * 2 / 25}}}
//                             ,{"mul_1_x", HOST, 1, {{"split_57_x", SIZE_OF_LAYER(57) * 2 / 25}}, {{"split_57_x", SIZE_OF_LAYER(57) * 2 / 25}}}
//                             ,{"mul_0_y", HOST, 1, {{"split_57_y", SIZE_OF_LAYER(57) * 2 / 25}}, {{"split_57_y", SIZE_OF_LAYER(57) * 2 / 25}}}
//                             ,{"pow_0", HOST, 1, {{"split_57_y", SIZE_OF_LAYER(57) * 2 / 25}}, {{"split_57_y", SIZE_OF_LAYER(57) * 2 / 25}}}
//                             ,{"mul_1_y", HOST, 1, {{"split_57_y", SIZE_OF_LAYER(57) * 2 / 25}, {"weights_mul_0", SIZE_OF_LAYER(57) * 2 / 25}}, {{"split_57_y", SIZE_OF_LAYER(57) * 2 / 25}}}
//                             ,{"csm_58", HOST,  20, {{"csm_50_out", SIZE_OF_LAYER(50)}}, {{"csm_58_out", SIZE_OF_LAYER(58)}}}
//                             ,{"transpose_1", HOST, 1, {{"csm_58_out", SIZE_OF_LAYER(58)}}, {{"csm_58_out_t", SIZE_OF_LAYER(58)}}}
//                             ,{"splitx3_1", HOST, 1, {{"csm_58_out_t", SIZE_OF_LAYER(58)}}, {{"split_58_x", SIZE_OF_LAYER(58) * 2 / 25}, {"split_58_y", SIZE_OF_LAYER(58) * 2 / 25}, {"split_58_z", SIZE_OF_LAYER(58) * 21 / 25}}}
//                             ,{"mul_2_x", HOST, 1, {{"split_58_x", SIZE_OF_LAYER(58) * 2 / 25}}, {{"split_58_x", SIZE_OF_LAYER(58) * 2 / 25}}}
//                             ,{"add_1", HOST, 1, {{"weights_add_1", SIZE_OF_LAYER(58) * 2 / 25}, {"split_58_x", SIZE_OF_LAYER(58) * 2 / 25}}, {{"split_58_x", SIZE_OF_LAYER(58) * 2 / 25}}}
//                             ,{"mul_3_x", HOST, 1, {{"split_58_x", SIZE_OF_LAYER(58) * 2 / 25}}, {{"split_58_x", SIZE_OF_LAYER(58) * 2 / 25}}}
//                             ,{"mul_2_y", HOST, 1, {{"split_58_y", SIZE_OF_LAYER(58) * 2 / 25}}, {{"split_58_y", SIZE_OF_LAYER(58) * 2 / 25}}}
//                             ,{"pow_1", HOST, 1, {{"split_58_y", SIZE_OF_LAYER(58) * 2 / 25}}, {{"split_58_y", SIZE_OF_LAYER(58) * 2 / 25}}}
//                             ,{"mul_3_y", HOST, 1, {{"split_58_y", SIZE_OF_LAYER(58) * 2 / 25}, {"weights_mul_1", SIZE_OF_LAYER(58) * 2 / 25}}, {{"split_58_y", SIZE_OF_LAYER(58) * 2 / 25}}}
//                             ,{"csm_59", HOST,  20, {{"csm_56_out", SIZE_OF_LAYER(56)}}, {{"csm_59_out", SIZE_OF_LAYER(59)}}}
//                             ,{"transpose_2", HOST, 1, {{"csm_59_out", SIZE_OF_LAYER(59)}}, {{"csm_59_out_t", SIZE_OF_LAYER(59)}}}
//                             ,{"splitx3_2", HOST, 1, {{"csm_59_out_t", SIZE_OF_LAYER(59)}}, {{"split_59_x", SIZE_OF_LAYER(59) * 2 / 25}, {"split_59_y", SIZE_OF_LAYER(59) * 2 / 25}, {"split_59_z", SIZE_OF_LAYER(59) * 21 / 25}}}
//                             ,{"mul_4_x", HOST, 1, {{"split_59_x", SIZE_OF_LAYER(59) * 2 / 25}}, {{"split_59_x", SIZE_OF_LAYER(59) * 2 / 25}}}
//                             ,{"add_2", HOST, 1, {{"weights_add_2", SIZE_OF_LAYER(59) * 2 / 25}, {"split_59_x", SIZE_OF_LAYER(59) * 2 / 25}}, {{"split_59_x", SIZE_OF_LAYER(59) * 2 / 25}}}
//                             ,{"mul_5_x", HOST, 1, {{"split_59_x", SIZE_OF_LAYER(59) * 2 / 25}}, {{"split_59_x", SIZE_OF_LAYER(59) * 2 / 25}}}
//                             ,{"mul_4_y", HOST, 1, {{"split_59_y", SIZE_OF_LAYER(59) * 2 / 25}}, {{"split_59_y", SIZE_OF_LAYER(59) * 2 / 25}}}
//                             ,{"pow_2", HOST, 1, {{"split_59_y", SIZE_OF_LAYER(59) * 2 / 25}}, {{"split_59_y", SIZE_OF_LAYER(59) * 2 / 25}}}
//                             ,{"mul_5_y", HOST, 1, {{"split_59_y", SIZE_OF_LAYER(59) * 2 / 25}, {"weights_mul_2", SIZE_OF_LAYER(59) * 2 / 25}}, {{"split_59_y", SIZE_OF_LAYER(59) * 2 / 25}}}
//                             ,{"detect", HOST, 1, {{"split_57_x", SIZE_OF_LAYER(57) * 2 / 25}, {"split_57_y", SIZE_OF_LAYER(57) * 2 / 25}, {"split_57_z", SIZE_OF_LAYER(57) * 21 / 25}
//                             , {"split_58_x", SIZE_OF_LAYER(58) * 2 / 25}, {"split_58_y", SIZE_OF_LAYER(58) * 2 / 25}, {"split_58_z", SIZE_OF_LAYER(58) * 21 / 25}
//                             , {"split_59_x", SIZE_OF_LAYER(59) * 2 / 25}, {"split_59_y", SIZE_OF_LAYER(59) * 2 / 25}, {"split_59_z", SIZE_OF_LAYER(59) * 21 / 25}}, {{"output", 0}}}
//                             // , {"detect", HOST, 1, {}, {{"output", 0}}}
//                             };

std::vector<Layer> yolov5n= {
                            {"mem_read_0", DEVICE, 0, {{"input", 640 * 640 * 3}}, {}}
                            ,{"mem_write_0", DEVICE, 0, {}, {{"csm_1_out", OUT_SIZE_OF_LAYER(1)}}}
                            // ,{"mem_read_1", DEVICE, 0, {{"csm_1_out", OUT_SIZE_OF_LAYER(1)}}, {}}
                            // ,{"mem_read_2", DEVICE, 0, {{"csm_1_out", OUT_SIZE_OF_LAYER(1)}}, {}}
                            // ,{"mem_write_1", DEVICE, 0, {}, {{"csm_7_out", OUT_SIZE_OF_LAYER(7)}}}
                            // ,{"mem_read_3", DEVICE, 0, {{"csm_7_out", OUT_SIZE_OF_LAYER(7)}}, {}}
                            // ,{"mem_read_4", DEVICE, 0, {{"csm_7_out", OUT_SIZE_OF_LAYER(7)}}, {}}
                            // ,{"mem_write_2", DEVICE, 0, {}, {{"csm_14_out", OUT_SIZE_OF_LAYER(14)}}}
                            // ,{"mem_write_3", DEVICE, 0, {}, {{"csm_15_out", OUT_SIZE_OF_LAYER(15)}}}
                            // ,{"mem_write_4", DEVICE, 0, {}, {{"add_4_out", HEIGHT_ADD_4 * WIDTH_ADD_4 * FILTERS_ADD_4}}}
                            // {"csm_0", HOST, 20, {{"input", 640*640*3}}, {{"csm_0_out", OUT_SIZE_OF_LAYER(0)}}}
                            // ,{"csm_1", HOST, 20, {{"csm_0_out", OUT_SIZE_OF_LAYER(0)}}, {{"csm_1_out", OUT_SIZE_OF_LAYER(1)}}}
                            // ,{"csm_2", HOST, 20, {{"csm_1_out", OUT_SIZE_OF_LAYER(1)}}, {{"csm_2_out", OUT_SIZE_OF_LAYER(2)}}}
                            // ,{"csm_3", HOST, 20, {{"csm_2_out", OUT_SIZE_OF_LAYER(2)}}, {{"csm_3_out", OUT_SIZE_OF_LAYER(3)}}}
                            // ,{"csm_4", HOST, 20, {{"csm_3_out", OUT_SIZE_OF_LAYER(3)}}, {{"csm_4_out", OUT_SIZE_OF_LAYER(4)}}}
                            // ,{"_add_0", HOST,  1, {{"csm_2_out", OUT_SIZE_OF_LAYER(2)}, {"csm_4_out", OUT_SIZE_OF_LAYER(4)}}, {{"csm_4_out", OUT_SIZE_OF_LAYER(4)}}}
                            // ,{"csm_5", HOST, 20, {{"csm_1_out", OUT_SIZE_OF_LAYER(1)}}, {{"csm_5_out", OUT_SIZE_OF_LAYER(5)}}}
                            // ,{"concat_0", HOST, 1, {{"csm_4_out", OUT_SIZE_OF_LAYER(4)}, {"csm_5_out", OUT_SIZE_OF_LAYER(5)}}, {{"concat_0_out", OUT_SIZE_OF_LAYER(4) + OUT_SIZE_OF_LAYER(5)}}}
                            // ,{"csm_6", HOST, 20, {{"concat_0_out", OUT_SIZE_OF_LAYER(4) + OUT_SIZE_OF_LAYER(5)}}, {{"csm_6_out", OUT_SIZE_OF_LAYER(6)}}}
                            // ,{"csm_7", HOST, 20, {{"csm_6_out", OUT_SIZE_OF_LAYER(6)}}, {{"csm_7_out", OUT_SIZE_OF_LAYER(7)}}}
                            // ,{"csm_8", HOST, 20, {{"csm_7_out", OUT_SIZE_OF_LAYER(7)}}, {{"csm_8_out", OUT_SIZE_OF_LAYER(8)}}}
                            // ,{"csm_9", HOST, 20, {{"csm_8_out", OUT_SIZE_OF_LAYER(8)}}, {{"csm_9_out", OUT_SIZE_OF_LAYER(9)}}}
                            // ,{"csm_10", HOST, 20, {{"csm_9_out", OUT_SIZE_OF_LAYER(9)}}, {{"csm_10_out", OUT_SIZE_OF_LAYER(10)}}}
                            // ,{"_add_1", HOST,  1, {{"csm_8_out", OUT_SIZE_OF_LAYER(8)}, {"csm_10_out", OUT_SIZE_OF_LAYER(10)}}, {{"csm_10_out", OUT_SIZE_OF_LAYER(10)}}}
                            // ,{"csm_11", HOST, 20, {{"csm_10_out", OUT_SIZE_OF_LAYER(10)}}, {{"csm_11_out", OUT_SIZE_OF_LAYER(11)}}}
                            // ,{"csm_12", HOST, 20, {{"csm_11_out", OUT_SIZE_OF_LAYER(11)}}, {{"csm_12_out", OUT_SIZE_OF_LAYER(12)}}}
                            // ,{"_add_2", HOST,  1, {{"csm_10_out", OUT_SIZE_OF_LAYER(10)}, {"csm_12_out", OUT_SIZE_OF_LAYER(12)}}, {{"csm_12_out", OUT_SIZE_OF_LAYER(12)}}}
                            // ,{"csm_13", HOST, 20, {{"csm_7_out", OUT_SIZE_OF_LAYER(7)}}, {{"csm_13_out", OUT_SIZE_OF_LAYER(13)}}}
                            // ,{"concat_1", HOST, 1, {{"csm_12_out", OUT_SIZE_OF_LAYER(12)}, {"csm_13_out", OUT_SIZE_OF_LAYER(13)}}, {{"concat_1_out", OUT_SIZE_OF_LAYER(12) + OUT_SIZE_OF_LAYER(13)}}}
                            // ,{"csm_14", HOST, 20, {{"concat_1_out", OUT_SIZE_OF_LAYER(12) + OUT_SIZE_OF_LAYER(13)}}, {{"csm_14_out", OUT_SIZE_OF_LAYER(14)}}}
                            // ,{"csm_15", HOST, 20, {{"csm_14_out", OUT_SIZE_OF_LAYER(14)}}, {{"csm_15_out", OUT_SIZE_OF_LAYER(15)}}}
                            // ,{"csm_16", HOST, 20, {{"csm_15_out", OUT_SIZE_OF_LAYER(15)}}, {{"csm_16_out", OUT_SIZE_OF_LAYER(16)}}}
                            // ,{"csm_17", HOST, 20, {{"csm_16_out", OUT_SIZE_OF_LAYER(16)}}, {{"csm_17_out", OUT_SIZE_OF_LAYER(17)}}}
                            // ,{"csm_18", HOST, 20, {{"csm_17_out", OUT_SIZE_OF_LAYER(17)}}, {{"csm_18_out", OUT_SIZE_OF_LAYER(18)}}}
                            // ,{"_add_3", HOST,  1, {{"csm_16_out", OUT_SIZE_OF_LAYER(16)}, {"csm_18_out", OUT_SIZE_OF_LAYER(18)}}, {{"csm_18_out", OUT_SIZE_OF_LAYER(18)}}}
                            // ,{"csm_19", HOST, 20, {{"csm_18_out", OUT_SIZE_OF_LAYER(18)}}, {{"csm_19_out", OUT_SIZE_OF_LAYER(19)}}}
                            // ,{"csm_20", HOST, 20, {{"csm_19_out", OUT_SIZE_OF_LAYER(19)}}, {{"csm_20_out", OUT_SIZE_OF_LAYER(20)}}}
                            // ,{"_add_4", HOST,  1, {{"csm_18_out", OUT_SIZE_OF_LAYER(18)}, {"csm_20_out", OUT_SIZE_OF_LAYER(20)}}, {{"add_4_out", HEIGHT_ADD_4 * WIDTH_ADD_4 * FILTERS_ADD_4}}}
                            // ,{"mem_read_5", DEVICE, 1, {{"add_4_out", HEIGHT_ADD_4 * WIDTH_ADD_4 * FILTERS_ADD_4}}, {}}
                            // ,{"mem_read_6", DEVICE, 1, {{"csm_15_out", OUT_SIZE_OF_LAYER(15)}}, {}}
                            // ,{"mem_write_5", DEVICE, 1, {}, {{"csm_24_out", OUT_SIZE_OF_LAYER(24)}}}
                            // ,{"mem_write_6", DEVICE, 1, {}, {{"csm_25_out", OUT_SIZE_OF_LAYER(25)}}}
                            // ,{"mem_read_7", DEVICE, 1, {{"csm_25_out", OUT_SIZE_OF_LAYER(25)}}, {}}
                            // ,{"mem_read_8", DEVICE, 1, {{"csm_25_out", OUT_SIZE_OF_LAYER(25)}}, {}}
                            // ,{"mem_write_7", DEVICE, 1, {}, {{"csm_31_out", OUT_SIZE_OF_LAYER(31)}}}
                            // ,{"csm_21", HOST, 20, {{"add_4_out", HEIGHT_ADD_4 * WIDTH_ADD_4 * FILTERS_ADD_4}}, {{"csm_21_out", OUT_SIZE_OF_LAYER(21)}}}
                            // ,{"csm_22", HOST, 20, {{"csm_21_out", OUT_SIZE_OF_LAYER(21)}}, {{"csm_22_out", OUT_SIZE_OF_LAYER(22)}}}
                            // ,{"_add_5", HOST,  1, {{"add_4_out", HEIGHT_ADD_4 * WIDTH_ADD_4 * FILTERS_ADD_4}, {"csm_22_out", OUT_SIZE_OF_LAYER(22)}}, {{"csm_22_out", OUT_SIZE_OF_LAYER(22)}}}
                            // ,{"csm_23", HOST, 20, {{"csm_15_out", OUT_SIZE_OF_LAYER(15)}}, {{"csm_23_out", OUT_SIZE_OF_LAYER(23)}}}
                            // ,{"concat_2", HOST, 1, {{"csm_22_out", OUT_SIZE_OF_LAYER(22)}, {"csm_23_out", OUT_SIZE_OF_LAYER(23)}}, {{"concat_2_out", OUT_SIZE_OF_LAYER(22) + OUT_SIZE_OF_LAYER(23)}}}
                            // ,{"csm_24", HOST, 20, {{"concat_2_out", OUT_SIZE_OF_LAYER(22) + OUT_SIZE_OF_LAYER(23)}}, {{"csm_24_out", OUT_SIZE_OF_LAYER(24)}}}
                            // ,{"csm_25", HOST, 20, {{"csm_24_out", OUT_SIZE_OF_LAYER(24)}}, {{"csm_25_out", OUT_SIZE_OF_LAYER(25)}}}
                            // ,{"csm_26", HOST, 20, {{"csm_25_out", OUT_SIZE_OF_LAYER(25)}}, {{"csm_26_out", OUT_SIZE_OF_LAYER(26)}}}
                            // ,{"csm_27", HOST, 20, {{"csm_26_out", OUT_SIZE_OF_LAYER(26)}}, {{"csm_27_out", OUT_SIZE_OF_LAYER(27)}}}
                            // ,{"csm_28", HOST, 20, {{"csm_27_out", OUT_SIZE_OF_LAYER(27)}}, {{"csm_28_out", OUT_SIZE_OF_LAYER(28)}}}
                            // ,{"_add_6", HOST,  1, {{"csm_26_out", OUT_SIZE_OF_LAYER(26)}, {"csm_28_out", OUT_SIZE_OF_LAYER(28)}}, {{"csm_28_out", OUT_SIZE_OF_LAYER(28)}}}
                            // ,{"csm_29", HOST, 20, {{"csm_25_out", OUT_SIZE_OF_LAYER(25)}}, {{"csm_29_out", OUT_SIZE_OF_LAYER(29)}}}
                            // ,{"concat_3", HOST, 1, {{"csm_28_out", OUT_SIZE_OF_LAYER(28)}, {"csm_29_out", OUT_SIZE_OF_LAYER(29)}}, {{"concat_3_out", OUT_SIZE_OF_LAYER(28) + OUT_SIZE_OF_LAYER(29)}}}
                            // ,{"csm_30", HOST, 20, {{"concat_3_out", OUT_SIZE_OF_LAYER(28) + OUT_SIZE_OF_LAYER(29)}}, {{"csm_30_out", OUT_SIZE_OF_LAYER(30)}}}
                            // ,{"csm_31", HOST, 20, {{"csm_30_out", OUT_SIZE_OF_LAYER(30)}}, {{"csm_31_out", OUT_SIZE_OF_LAYER(31)}}}
                            // ,{"maxpool_0", HOST, 20, {{"csm_31_out", OUT_SIZE_OF_LAYER(31)}}, {{"maxpool_0_out", 128 * 20 * 20}}}
                            // ,{"maxpool_1", HOST, 20, {{"maxpool_0_out", 128 * 20 * 20}}, {{"maxpool_1_out", 128 * 20 * 20}}}
                            // ,{"maxpool_2", HOST, 20, {{"maxpool_1_out", 128 * 20 * 20}}, {{"maxpool_2_out", 128 * 20 * 20}}}
                            // ,{"concat_4_1", HOST, 1, {{"csm_31_out", OUT_SIZE_OF_LAYER(31)}, {"maxpool_0_out", 128 * 20 * 20}}, {{"concat_4_1_out", OUT_SIZE_OF_LAYER(31) + 128 * 20 * 20}}}
                            // ,{"concat_4_2", HOST, 1, {{"concat_4_1_out", OUT_SIZE_OF_LAYER(31) + 128 * 20 * 20}, {"maxpool_1_out", 128 * 20 * 20}}, {{"concat_4_2_out", OUT_SIZE_OF_LAYER(31) + 128 * 20 * 20 * 2}}}
                            // ,{"concat_4_3", HOST, 1, {{"concat_4_2_out", OUT_SIZE_OF_LAYER(31) + 128 * 20 * 20 * 2}, {"maxpool_2_out", 128 * 20 * 20}}, {{"concat_4_3_out", OUT_SIZE_OF_LAYER(31) + 128 * 20 * 20 * 3}}}
                            // ,{"csm_32", HOST,  20, {{"concat_4_3_out", 20 * 20 * 512}}, {{"csm_32_out", OUT_SIZE_OF_LAYER(32)}}}
                            // ,{"csm_33", HOST,  20, {{"csm_32_out", OUT_SIZE_OF_LAYER(32)}}, {{"csm_33_out", OUT_SIZE_OF_LAYER(33)}}}
                            // ,{"resize_0", HOST, 1, {{"csm_33_out", OUT_SIZE_OF_LAYER(33)}}, {{"resize_0_output", OUT_SIZE_OF_LAYER(33) * 4}}}
                            // ,{"concat_5", HOST, 1, {{"resize_0_output", OUT_SIZE_OF_LAYER(33) * 4}, {"csm_24_out", OUT_SIZE_OF_LAYER(24)}}, {{"concat_5_out", OUT_SIZE_OF_LAYER(33) * 4 + OUT_SIZE_OF_LAYER(24)}}}
                            // ,{"csm_34", HOST,  20, {{"concat_5_out", OUT_SIZE_OF_LAYER(33) + OUT_SIZE_OF_LAYER(24)}}, {{"csm_34_out", OUT_SIZE_OF_LAYER(34)}}}
                            // ,{"csm_35", HOST,  20, {{"csm_34_out", OUT_SIZE_OF_LAYER(34)}}, {{"csm_35_out", OUT_SIZE_OF_LAYER(35)}}}
                            // ,{"csm_36", HOST,  20, {{"csm_35_out", OUT_SIZE_OF_LAYER(35)}}, {{"csm_36_out", OUT_SIZE_OF_LAYER(36)}}}
                            // ,{"csm_37", HOST,  20, {{"concat_5_out", OUT_SIZE_OF_LAYER(33) + OUT_SIZE_OF_LAYER(24)}}, {{"csm_37_out", OUT_SIZE_OF_LAYER(37)}}}
                            // ,{"concat_6", HOST, 1, {{"csm_36_out", OUT_SIZE_OF_LAYER(36)}, {"csm_37_out", OUT_SIZE_OF_LAYER(37)}}, {{"concat_6_out", OUT_SIZE_OF_LAYER(36) + OUT_SIZE_OF_LAYER(37)}}}
                            // ,{"csm_38", HOST,  20, {{"concat_6_out", OUT_SIZE_OF_LAYER(36) + OUT_SIZE_OF_LAYER(37)}}, {{"csm_38_out", OUT_SIZE_OF_LAYER(38)}}}
                            // ,{"csm_38", HOST,  20, {{"concat_6_out", OUT_SIZE_OF_LAYER(36) + OUT_SIZE_OF_LAYER(37)}}, {{"csm_38_out", OUT_SIZE_OF_LAYER(38)}}}
                            // ,{"csm_39", HOST,  20, {{"csm_38_out", OUT_SIZE_OF_LAYER(38)}}, {{"csm_39_out", OUT_SIZE_OF_LAYER(39)}}}
                            // ,{"resize_1", HOST, 1, {{"csm_39_out", OUT_SIZE_OF_LAYER(39)}}, {{"resize_1_output", OUT_SIZE_OF_LAYER(39) * 4}}}
                            // ,{"concat_7", HOST, 1, {{"resize_1_output", OUT_SIZE_OF_LAYER(39) * 4}, {"csm_14_out", OUT_SIZE_OF_LAYER(14)}}, {{"concat_7_out", OUT_SIZE_OF_LAYER(39) * 4 + OUT_SIZE_OF_LAYER(14)}}}
                            // ,{"csm_40", HOST,  20, {{"concat_7_out", OUT_SIZE_OF_LAYER(39) + OUT_SIZE_OF_LAYER(14)}}, {{"csm_40_out", OUT_SIZE_OF_LAYER(40)}}}
                            // ,{"csm_41", HOST,  20, {{"csm_40_out", OUT_SIZE_OF_LAYER(40)}}, {{"csm_41_out", OUT_SIZE_OF_LAYER(41)}}}
                            // ,{"csm_42", HOST,  20, {{"csm_41_out", OUT_SIZE_OF_LAYER(41)}}, {{"csm_42_out", OUT_SIZE_OF_LAYER(42)}}}
                            // ,{"csm_43", HOST,  20, {{"concat_7_out", OUT_SIZE_OF_LAYER(39) + OUT_SIZE_OF_LAYER(14)}}, {{"csm_43_out", OUT_SIZE_OF_LAYER(43)}}}
                            // ,{"concat_8", HOST, 1, {{"csm_42_out", OUT_SIZE_OF_LAYER(42)}, {"csm_43_out", OUT_SIZE_OF_LAYER(43)}}, {{"concat_8_out", OUT_SIZE_OF_LAYER(42) + OUT_SIZE_OF_LAYER(43)}}}
                            // ,{"csm_44", HOST,  20, {{"concat_8_out", OUT_SIZE_OF_LAYER(42) + OUT_SIZE_OF_LAYER(43)}}, {{"csm_44_out", OUT_SIZE_OF_LAYER(44)}}}
                            // ,{"csm_45", HOST,  20, {{"csm_44_out", OUT_SIZE_OF_LAYER(44)}}, {{"csm_45_out", OUT_SIZE_OF_LAYER(45)}}}
                            // ,{"concat_9", HOST, 1, {{"csm_45_out", OUT_SIZE_OF_LAYER(45)}, {"csm_39_out", OUT_SIZE_OF_LAYER(39)}}, {{"concat_9_out", OUT_SIZE_OF_LAYER(45) + OUT_SIZE_OF_LAYER(39)}}}
                            // ,{"csm_46", HOST,  20, {{"concat_9_out", OUT_SIZE_OF_LAYER(45) + OUT_SIZE_OF_LAYER(39)}}, {{"csm_46_out", OUT_SIZE_OF_LAYER(46)}}}
                            // ,{"csm_47", HOST,  20, {{"csm_46_out", OUT_SIZE_OF_LAYER(46)}}, {{"csm_47_out", OUT_SIZE_OF_LAYER(47)}}}
                            // ,{"csm_48", HOST,  20, {{"csm_47_out", OUT_SIZE_OF_LAYER(47)}}, {{"csm_48_out", OUT_SIZE_OF_LAYER(48)}}}
                            // ,{"csm_49", HOST,  20, {{"concat_9_out", OUT_SIZE_OF_LAYER(45) + OUT_SIZE_OF_LAYER(39)}}, {{"csm_49_out", OUT_SIZE_OF_LAYER(49)}}}
                            // ,{"concat_10", HOST, 1, {{"csm_48_out", OUT_SIZE_OF_LAYER(48)}, {"csm_49_out", OUT_SIZE_OF_LAYER(49)}}, {{"concat_10_out", OUT_SIZE_OF_LAYER(48) + OUT_SIZE_OF_LAYER(49)}}}
                            // ,{"csm_50", HOST,  20, {{"concat_10_out", OUT_SIZE_OF_LAYER(49) + OUT_SIZE_OF_LAYER(49)}}, {{"csm_50_out", OUT_SIZE_OF_LAYER(50)}}}
                            // ,{"csm_51", HOST,  20, {{"csm_50_out", OUT_SIZE_OF_LAYER(50)}}, {{"csm_51_out", OUT_SIZE_OF_LAYER(51)}}}
                            // ,{"concat_11", HOST, 1, {{"csm_51_out", OUT_SIZE_OF_LAYER(51)}, {"csm_33_out", OUT_SIZE_OF_LAYER(33)}}, {{"concat_11_out", OUT_SIZE_OF_LAYER(51) + OUT_SIZE_OF_LAYER(33)}}}
                            // ,{"csm_52", HOST,  20, {{"concat_11_out", OUT_SIZE_OF_LAYER(51) + OUT_SIZE_OF_LAYER(33)}}, {{"csm_52_out", OUT_SIZE_OF_LAYER(52)}}}
                            // ,{"csm_53", HOST,  20, {{"csm_52_out", OUT_SIZE_OF_LAYER(52)}}, {{"csm_53_out", OUT_SIZE_OF_LAYER(53)}}}
                            // ,{"csm_54", HOST,  20, {{"csm_53_out", OUT_SIZE_OF_LAYER(53)}}, {{"csm_54_out", OUT_SIZE_OF_LAYER(54)}}}
                            // ,{"csm_55", HOST,  20, {{"concat_11_out", OUT_SIZE_OF_LAYER(51) + OUT_SIZE_OF_LAYER(33)}}, {{"csm_55_out", OUT_SIZE_OF_LAYER(55)}}}
                            // ,{"concat_12", HOST, 1, {{"csm_54_out", OUT_SIZE_OF_LAYER(54)}, {"csm_55_out", OUT_SIZE_OF_LAYER(55)}}, {{"concat_12_out", OUT_SIZE_OF_LAYER(54) + OUT_SIZE_OF_LAYER(55)}}}
                            // ,{"csm_56", HOST,  20, {{"concat_12_out", OUT_SIZE_OF_LAYER(54) + OUT_SIZE_OF_LAYER(55)}}, {{"csm_56_out", OUT_SIZE_OF_LAYER(56)}}}
                            // ,{"csm_57", HOST,  20, {{"csm_44_out", OUT_SIZE_OF_LAYER(44)}}, {{"csm_57_out", OUT_SIZE_OF_LAYER(57)}}}
                            // ,{"transpose_0", HOST, 1, {{"csm_57_out", OUT_SIZE_OF_LAYER(57)}}, {{"csm_57_out_t", OUT_SIZE_OF_LAYER(57)}}}
                            // ,{"splitx3_0", HOST, 1, {{"csm_57_out_t", OUT_SIZE_OF_LAYER(57)}}, {{"split_57_x", OUT_SIZE_OF_LAYER(57) * 2 / 25}, {"split_57_y", OUT_SIZE_OF_LAYER(57) * 2 / 25}, {"split_57_z", OUT_SIZE_OF_LAYER(57) * 21 / 25}}}
                            // ,{"mul_0_x", HOST, 1, {{"split_57_x", OUT_SIZE_OF_LAYER(57) * 2 / 25}}, {{"split_57_x", OUT_SIZE_OF_LAYER(57) * 2 / 25}}}
                            // ,{"add_0", HOST, 1, {{"weights_add_0", OUT_SIZE_OF_LAYER(57) * 2 / 25}, {"split_57_x", OUT_SIZE_OF_LAYER(57) * 2 / 25}}, {{"split_57_x", OUT_SIZE_OF_LAYER(57) * 2 / 25}}}
                            // ,{"mul_1_x", HOST, 1, {{"split_57_x", OUT_SIZE_OF_LAYER(57) * 2 / 25}}, {{"split_57_x", OUT_SIZE_OF_LAYER(57) * 2 / 25}}}
                            // ,{"mul_0_y", HOST, 1, {{"split_57_y", OUT_SIZE_OF_LAYER(57) * 2 / 25}}, {{"split_57_y", OUT_SIZE_OF_LAYER(57) * 2 / 25}}}
                            // ,{"pow_0", HOST, 1, {{"split_57_y", OUT_SIZE_OF_LAYER(57) * 2 / 25}}, {{"split_57_y", OUT_SIZE_OF_LAYER(57) * 2 / 25}}}
                            // ,{"mul_1_y", HOST, 1, {{"split_57_y", OUT_SIZE_OF_LAYER(57) * 2 / 25}, {"weights_mul_0", OUT_SIZE_OF_LAYER(57) * 2 / 25}}, {{"split_57_y", OUT_SIZE_OF_LAYER(57) * 2 / 25}}}
                            // ,{"csm_58", HOST,  20, {{"csm_50_out", OUT_SIZE_OF_LAYER(50)}}, {{"csm_58_out", OUT_SIZE_OF_LAYER(58)}}}
                            // ,{"transpose_1", HOST, 1, {{"csm_58_out", OUT_SIZE_OF_LAYER(58)}}, {{"csm_58_out_t", OUT_SIZE_OF_LAYER(58)}}}
                            // ,{"splitx3_1", HOST, 1, {{"csm_58_out_t", OUT_SIZE_OF_LAYER(58)}}, {{"split_58_x", OUT_SIZE_OF_LAYER(58) * 2 / 25}, {"split_58_y", OUT_SIZE_OF_LAYER(58) * 2 / 25}, {"split_58_z", OUT_SIZE_OF_LAYER(58) * 21 / 25}}}
                            // ,{"mul_2_x", HOST, 1, {{"split_58_x", OUT_SIZE_OF_LAYER(58) * 2 / 25}}, {{"split_58_x", OUT_SIZE_OF_LAYER(58) * 2 / 25}}}
                            // ,{"add_1", HOST, 1, {{"weights_add_1", OUT_SIZE_OF_LAYER(58) * 2 / 25}, {"split_58_x", OUT_SIZE_OF_LAYER(58) * 2 / 25}}, {{"split_58_x", OUT_SIZE_OF_LAYER(58) * 2 / 25}}}
                            // ,{"mul_3_x", HOST, 1, {{"split_58_x", OUT_SIZE_OF_LAYER(58) * 2 / 25}}, {{"split_58_x", OUT_SIZE_OF_LAYER(58) * 2 / 25}}}
                            // ,{"mul_2_y", HOST, 1, {{"split_58_y", OUT_SIZE_OF_LAYER(58) * 2 / 25}}, {{"split_58_y", OUT_SIZE_OF_LAYER(58) * 2 / 25}}}
                            // ,{"pow_1", HOST, 1, {{"split_58_y", OUT_SIZE_OF_LAYER(58) * 2 / 25}}, {{"split_58_y", OUT_SIZE_OF_LAYER(58) * 2 / 25}}}
                            // ,{"mul_3_y", HOST, 1, {{"split_58_y", OUT_SIZE_OF_LAYER(58) * 2 / 25}, {"weights_mul_1", OUT_SIZE_OF_LAYER(58) * 2 / 25}}, {{"split_58_y", OUT_SIZE_OF_LAYER(58) * 2 / 25}}}
                            // ,{"csm_59", HOST,  20, {{"csm_56_out", OUT_SIZE_OF_LAYER(56)}}, {{"csm_59_out", OUT_SIZE_OF_LAYER(59)}}}
                            // ,{"transpose_2", HOST, 1, {{"csm_59_out", OUT_SIZE_OF_LAYER(59)}}, {{"csm_59_out_t", OUT_SIZE_OF_LAYER(59)}}}
                            // ,{"splitx3_2", HOST, 1, {{"csm_59_out_t", OUT_SIZE_OF_LAYER(59)}}, {{"split_59_x", OUT_SIZE_OF_LAYER(59) * 2 / 25}, {"split_59_y", OUT_SIZE_OF_LAYER(59) * 2 / 25}, {"split_59_z", OUT_SIZE_OF_LAYER(59) * 21 / 25}}}
                            // ,{"mul_4_x", HOST, 1, {{"split_59_x", OUT_SIZE_OF_LAYER(59) * 2 / 25}}, {{"split_59_x", OUT_SIZE_OF_LAYER(59) * 2 / 25}}}
                            // ,{"add_2", HOST, 1, {{"weights_add_2", OUT_SIZE_OF_LAYER(59) * 2 / 25}, {"split_59_x", OUT_SIZE_OF_LAYER(59) * 2 / 25}}, {{"split_59_x", OUT_SIZE_OF_LAYER(59) * 2 / 25}}}
                            // ,{"mul_5_x", HOST, 1, {{"split_59_x", OUT_SIZE_OF_LAYER(59) * 2 / 25}}, {{"split_59_x", OUT_SIZE_OF_LAYER(59) * 2 / 25}}}
                            // ,{"mul_4_y", HOST, 1, {{"split_59_y", OUT_SIZE_OF_LAYER(59) * 2 / 25}}, {{"split_59_y", OUT_SIZE_OF_LAYER(59) * 2 / 25}}}
                            // ,{"pow_2", HOST, 1, {{"split_59_y", OUT_SIZE_OF_LAYER(59) * 2 / 25}}, {{"split_59_y", OUT_SIZE_OF_LAYER(59) * 2 / 25}}}
                            // ,{"mul_5_y", HOST, 1, {{"split_59_y", OUT_SIZE_OF_LAYER(59) * 2 / 25}, {"weights_mul_2", OUT_SIZE_OF_LAYER(59) * 2 / 25}}, {{"split_59_y", OUT_SIZE_OF_LAYER(59) * 2 / 25}}}
                            // ,{"detect", HOST, 1, {{"split_57_x", OUT_SIZE_OF_LAYER(57) * 2 / 25}, {"split_57_y", OUT_SIZE_OF_LAYER(57) * 2 / 25}, {"split_57_z", OUT_SIZE_OF_LAYER(57) * 21 / 25}
                            // , {"split_58_x", OUT_SIZE_OF_LAYER(58) * 2 / 25}, {"split_58_y", OUT_SIZE_OF_LAYER(58) * 2 / 25}, {"split_58_z", OUT_SIZE_OF_LAYER(58) * 21 / 25}
                            // , {"split_59_x", OUT_SIZE_OF_LAYER(59) * 2 / 25}, {"split_59_y", OUT_SIZE_OF_LAYER(59) * 2 / 25}, {"split_59_z", OUT_SIZE_OF_LAYER(59) * 21 / 25}}, {{"output", 0}}}
                            , {"detect", HOST, 1, {{"csm_1_out", OUT_SIZE_OF_LAYER(1)}}, {{"output", 0}}}
                            };


std::unordered_map<std::string, void(*)(int, int, std::vector<std::reference_wrapper<std::vector<int8_t, AlignedAllocator<int8_t, 64>>>> , std::vector<std::reference_wrapper<std::vector<int8_t, AlignedAllocator<int8_t, 64>>>>)> CPU_functions{
    {"csm_0", REGISTET_CSM(0, int8_t, int)}
    ,{"csm_1", REGISTET_CSM(1, int8_t, int)}
    ,{"csm_2", REGISTET_CSM(2, int8_t, int)}
    ,{"csm_3", REGISTET_CSM(3, int8_t, int)}
    ,{"csm_4", REGISTET_CSM(4, int8_t, int)}
    ,{"csm_5", REGISTET_CSM(5, int8_t, int)}
    ,{"csm_6", REGISTET_CSM(6, int8_t, int)}
    ,{"csm_7", REGISTET_CSM(7, int8_t, int)}
    ,{"csm_8", REGISTET_CSM(8, int8_t, int)}
    ,{"csm_9", REGISTET_CSM(9, int8_t, int)}
    ,{"csm_10", REGISTET_CSM(10, int8_t, int)}
    ,{"csm_11", REGISTET_CSM(11, int8_t, int)}
    ,{"csm_12", REGISTET_CSM(12, int8_t, int)}
    ,{"csm_13", REGISTET_CSM(13, int8_t, int)}
    ,{"csm_14", REGISTET_CSM(14, int8_t, int)}
    ,{"csm_15", REGISTET_CSM(15, int8_t, int)}
    ,{"csm_16", REGISTET_CSM(16, int8_t, int)}
    ,{"csm_17", REGISTET_CSM(17, int8_t, int)}
    ,{"csm_18", REGISTET_CSM(18, int8_t, int)}
    ,{"csm_19", REGISTET_CSM(19, int8_t, int)}
    ,{"csm_20", REGISTET_CSM(20, int8_t, int)}
    ,{"csm_21", REGISTET_CSM(21, int8_t, int)}
    ,{"csm_22", REGISTET_CSM(22, int8_t, int)}
    ,{"csm_23", REGISTET_CSM(23, int8_t, int)}
    ,{"csm_24", REGISTET_CSM(24, int8_t, int)}
    ,{"csm_25", REGISTET_CSM(25, int8_t, int)}
    ,{"csm_26", REGISTET_CSM(26, int8_t, int)}
    ,{"csm_27", REGISTET_CSM(27, int8_t, int)}
    ,{"csm_28", REGISTET_CSM(28, int8_t, int)}
    ,{"csm_29", REGISTET_CSM(29, int8_t, int)}
    ,{"csm_30", REGISTET_CSM(30, int8_t, int)}
    ,{"csm_31", REGISTET_CSM(31, int8_t, int)}
    ,{"maxpool_0", REGISTET_MAXPOOL(0, int8_t)}
    ,{"maxpool_1", REGISTET_MAXPOOL(1, int8_t)}
    ,{"maxpool_2", REGISTET_MAXPOOL(2, int8_t)}
    ,{"concat_0", REGISTET_CONCAT(0, int8_t)}
    ,{"concat_1", REGISTET_CONCAT(1, int8_t)}
    ,{"concat_2", REGISTET_CONCAT(2, int8_t)}
    ,{"concat_3", REGISTET_CONCAT(3, int8_t)}
    ,{"concat_4_1", REGISTET_CONCAT_WITHOUT_QUANT(4_1, int8_t)}
    ,{"concat_4_2", REGISTET_CONCAT_WITHOUT_QUANT(4_2, int8_t)}
    ,{"concat_4_3", REGISTET_CONCAT_WITHOUT_QUANT(4_3, int8_t)}
    ,{"csm_32", REGISTET_CSM(32, int8_t, int)}
    ,{"csm_33", REGISTET_CSM(33, int8_t, int)}
    ,{"resize_0", REGISTET_RESIZE(33, int8_t)}
    ,{"resize_1", REGISTET_RESIZE(39, int8_t)}
    ,{"concat_5", REGISTET_CONCAT(5, int8_t)}
    ,{"csm_34", REGISTET_CSM(34, int8_t, int)}
    ,{"csm_35", REGISTET_CSM(35, int8_t, int)}
    ,{"csm_36", REGISTET_CSM(36, int8_t, int)}
    ,{"csm_37", REGISTET_CSM(37, int8_t, int)}
    ,{"csm_38", REGISTET_CSM(38, int8_t, int)}
    ,{"csm_39", REGISTET_CSM(39, int8_t, int)}
    ,{"csm_40", REGISTET_CSM(40, int8_t, int)}
    ,{"csm_41", REGISTET_CSM(41, int8_t, int)}
    ,{"csm_42", REGISTET_CSM(42, int8_t, int)}
    ,{"csm_43", REGISTET_CSM(43, int8_t, int)}
    ,{"csm_44", REGISTET_CSM(44, int8_t, int)}
    ,{"csm_45", REGISTET_CSM(45, int8_t, int)}
    ,{"csm_46", REGISTET_CSM(46, int8_t, int)}
    ,{"csm_47", REGISTET_CSM(47, int8_t, int)}
    ,{"csm_48", REGISTET_CSM(48, int8_t, int)}
    ,{"csm_49", REGISTET_CSM(49, int8_t, int)}
    ,{"csm_50", REGISTET_CSM(50, int8_t, int)}
    ,{"csm_51", REGISTET_CSM(51, int8_t, int)}
    ,{"csm_52", REGISTET_CSM(52, int8_t, int)}
    ,{"csm_53", REGISTET_CSM(53, int8_t, int)}
    ,{"csm_54", REGISTET_CSM(54, int8_t, int)}
    ,{"csm_55", REGISTET_CSM(55, int8_t, int)}
    ,{"csm_56", REGISTET_CSM(56, int8_t, int)}
    ,{"csm_57", REGISTET_CSM_LAST(57, int8_t, int)}
    ,{"csm_58", REGISTET_CSM_LAST(58, int8_t, int)}
    ,{"csm_59", REGISTET_CSM_LAST(59, int8_t, int)}
    ,{"concat_6", REGISTET_CONCAT(6, int8_t)}
    ,{"concat_7", REGISTET_CONCAT(7, int8_t)}
    ,{"concat_8", REGISTET_CONCAT(8, int8_t)}
    ,{"concat_9", REGISTET_CONCAT(9, int8_t)}
    ,{"concat_10", REGISTET_CONCAT(10, int8_t)}
    ,{"concat_11", REGISTET_CONCAT(11, int8_t)}
    ,{"concat_12", REGISTET_CONCAT(12, int8_t)}
    ,{"transpose_0", REGISTET_TRANSPOSE(57, int8_t, 3)}
    ,{"transpose_1", REGISTET_TRANSPOSE(58, int8_t, 3)}
    ,{"transpose_2", REGISTET_TRANSPOSE(59, int8_t, 3)}
    ,{"splitx3_0", REGISTET_SPLITX3(57, int8_t, 2, 2, 21)}
    ,{"splitx3_1", REGISTET_SPLITX3(58, int8_t, 2, 2, 21)}
    ,{"splitx3_2", REGISTET_SPLITX3(59, int8_t, 2, 2, 21)}
    ,{"mul_0_x", REGISTET_MUL_CONST(0, int8_t, 127, x)}
    ,{"mul_1_x", REGISTET_MUL_CONST(1, int8_t, 127, x)}
    ,{"mul_2_x", REGISTET_MUL_CONST(2, int8_t, 127, x)}
    ,{"mul_3_x", REGISTET_MUL_CONST(3, int8_t, 127, x)}
    ,{"mul_4_x", REGISTET_MUL_CONST(4, int8_t, 127, x)}
    ,{"mul_5_x", REGISTET_MUL_CONST(5, int8_t, 127, x)}
    ,{"mul_0_y", REGISTET_MUL_CONST(0, int8_t, 127, y)}
    ,{"mul_1_y", REGISTET_MUL(1, int8_t, y)}
    ,{"mul_2_y", REGISTET_MUL_CONST(2, int8_t, 127, y)}
    ,{"mul_3_y", REGISTET_MUL(3, int8_t, y)}
    ,{"mul_4_y", REGISTET_MUL_CONST(4, int8_t, 127, y)}
    ,{"mul_5_y", REGISTET_MUL(5, int8_t, y)}
    ,{"_add_0", REGISTET_ADD(0, int8_t)}
    ,{"_add_1", REGISTET_ADD(1, int8_t)}
    ,{"_add_2", REGISTET_ADD(2, int8_t)}
    ,{"_add_3", REGISTET_ADD(3, int8_t)}
    ,{"_add_4", REGISTET_ADD(4, int8_t)}
    ,{"_add_5", REGISTET_ADD(5, int8_t)}
    ,{"_add_6", REGISTET_ADD(6, int8_t)}
    ,{"add_0", REGISTET_ADD_WITH_XY(0, int8_t, x)}
    ,{"add_1", REGISTET_ADD_WITH_XY(1, int8_t, x)}
    ,{"add_2", REGISTET_ADD_WITH_XY(2, int8_t, x)}
    ,{"pow_0", pow<int8_t, OUTPUT_CHANNELS_LAYER_57 * 2 / 25, OUTPUT_SIZE_LAYER_57, OUTPUT_SIZE_LAYER_57, pow_57_split>}
    ,{"pow_1", pow<int8_t, OUTPUT_CHANNELS_LAYER_58 * 2 / 25, OUTPUT_SIZE_LAYER_58, OUTPUT_SIZE_LAYER_58, pow_58_split>}
    ,{"pow_2", pow<int8_t, OUTPUT_CHANNELS_LAYER_59 * 2 / 25, OUTPUT_SIZE_LAYER_59, OUTPUT_SIZE_LAYER_59, pow_59_split>}
};

std::vector<Pre_Layer> yolov5n_pre= {
                            // {"conv_ld_wt_0", DEVICE, {{"weights_0", 1728}}}
                            // ,{"conv_ld_wt_1", DEVICE, {{"weights_1", 4608}}}
                            // ,{"conv_ld_wt_2", DEVICE, {{"weights_2", 512}}}
                            // ,{"conv_ld_wt_3", DEVICE, {{"weights_3", 256}}}
                            // ,{"conv_ld_wt_4", DEVICE, {{"weights_4", 2304}}}
                            // ,{"conv_ld_wt_5", DEVICE, {{"weights_5", 512}}}
                            // ,{"conv_ld_wt_6", DEVICE, {{"weights_6", 1024}}}
                            // ,{"conv_ld_wt_7", DEVICE, {{"weights_7", 18432}}}
                            // ,{"conv_ld_wt_8", DEVICE, {{"weights_8", 2048}}}
                            // ,{"conv_ld_wt_9", DEVICE, {{"weights_9", 1024}}}
                            // ,{"conv_ld_wt_10", DEVICE, {{"weights_10", 9216}}}
                            // ,{"conv_ld_wt_11", DEVICE, {{"weights_11", 1024}}}
                            // ,{"conv_ld_wt_12", DEVICE, {{"weights_12", 9216}}}
                            // ,{"conv_ld_wt_13", DEVICE, {{"weights_13", 2048}}}
                            // ,{"conv_ld_wt_14", DEVICE, {{"weights_14", 4096}}}
                            // ,{"conv_ld_wt_15", DEVICE, {{"weights_15", 73728}}}
                            // ,{"conv_ld_wt_16", DEVICE, {{"weights_16", 8192}}}
                            // ,{"conv_ld_wt_17", DEVICE, {{"weights_17", 4096}}}
                            // ,{"conv_ld_wt_18", DEVICE, {{"weights_18", 36864}}}
                            // ,{"conv_ld_wt_19", DEVICE, {{"weights_19", 4096}}}
                            // ,{"conv_ld_wt_20", DEVICE, {{"weights_20", 36864}}}
                            // ,{"conv_ld_wt_21", DEVICE, {{"weights_21", 4096}}}
                            // ,{"conv_ld_wt_22", DEVICE, {{"weights_22", 36864}}}
                            // ,{"conv_ld_wt_23", DEVICE, {{"weights_23", 8192}}}
                            };
const std::unordered_map<std::string, std::vector<int8_t, AlignedAllocator<int8_t, 64>>> yolov5n_weights{
                                                                    // {"weights_0", weights_0}
                                                                    // ,{"weights_1", weights_1}
                                                                    // ,{"weights_2", weights_2}
                                                                    // ,{"weights_3", weights_3}
                                                                    // ,{"weights_4", weights_4}
                                                                    // ,{"weights_5", weights_5}
                                                                    // ,{"weights_6", weights_6}
                                                                    // ,{"weights_7", weights_7}
                                                                    // ,{"weights_8", weights_8}
                                                                    // ,{"weights_9", weights_9}
                                                                    // ,{"weights_10", weights_10}
                                                                    // ,{"weights_11", weights_11}
                                                                    // ,{"weights_12", weights_12}
                                                                    // ,{"weights_13", weights_13}
                                                                    // ,{"weights_14", weights_14}
                                                                    // ,{"weights_15", weights_15}
                                                                    // ,{"weights_16", weights_16}
                                                                    // ,{"weights_17", weights_17}
                                                                    // ,{"weights_18", weights_18}
                                                                    // ,{"weights_19", weights_19}
                                                                    // ,{"weights_20", weights_20}
                                                                    // ,{"weights_21", weights_21}
                                                                    // ,{"weights_22", weights_22}
                                                                    // ,{"weights_23", weights_23}
                                                                    // ,{"weights_24", weights_24}
                                                                    // ,{"weights_25", weights_25}
                                                                    // ,{"weights_26", weights_26}
                                                                    // ,{"weights_27", weights_27}
                                                                    // ,{"weights_28", weights_28}
                                                                    // ,{"weights_29", weights_29}
                                                                    // ,{"weights_30", weights_30}
                                                                    // ,{"weights_31", weights_31}
                                                                    // ,{"weights_32", weights_32}
                                                                    // ,{"weights_33", weights_33}
                                                                    // ,{"weights_34", weights_34}
                                                                    // ,{"weights_35", weights_35}
                                                                    // ,{"weights_36", weights_36}
                                                                    // ,{"weights_37", weights_37}
                                                                    // ,{"weights_38", weights_38}
                                                                    // ,{"weights_39", weights_39}
                                                                    // ,{"weights_40", weights_40}
                                                                    // ,{"weights_41", weights_41}
                                                                    // ,{"weights_42", weights_42}
                                                                    // ,{"weights_43", weights_43}
                                                                    // ,{"weights_44", weights_44}
                                                                    // ,{"weights_45", weights_45}
                                                                    // ,{"weights_46", weights_46}
                                                                    // ,{"weights_47", weights_47}
                                                                    // ,{"weights_48", weights_48}
                                                                    // ,{"weights_49", weights_49}
                                                                    // ,{"weights_50", weights_50}
                                                                    // ,{"weights_51", weights_51}
                                                                    // ,{"weights_52", weights_52}
                                                                    // ,{"weights_53", weights_53}
                                                                    // ,{"weights_54", weights_54}
                                                                    // ,{"weights_55", weights_55}
                                                                    // ,{"weights_56", weights_56}
                                                                    // ,{"weights_57", weights_57}
                                                                    // ,{"weights_58", weights_58}
                                                                    // ,{"weights_59", weights_59}
                                                                    {"weights_add_0", weights_add_0}
                                                                    ,{"weights_add_1", weights_add_1}
                                                                    ,{"weights_add_2", weights_add_2}
                                                                    ,{"weights_mul_0", weights_mul_0}
                                                                    ,{"weights_mul_1", weights_mul_1}
                                                                    ,{"weights_mul_2", weights_mul_2}
                                                                    };
void testlastfunction(std::vector<std::reference_wrapper<std::vector<int8_t, AlignedAllocator<int8_t, 64>>>>, std::vector<Box>&){
    return;
}
// void(* last_function)(std::vector<std::reference_wrapper<std::vector<int8_t, AlignedAllocator<int8_t, 64>>>>, std::vector<Box>&) = detect<
//                             int8_t
//                             ,25200
//                             ,OUTPUT_SIZE_LAYER_57
//                             ,OUTPUT_SIZE_LAYER_58
//                             ,OUTPUT_SIZE_LAYER_59
//                             ,OUTPUT_SIZE_LAYER_57
//                             ,OUTPUT_SIZE_LAYER_58
//                             ,OUTPUT_SIZE_LAYER_59
//                             ,2
//                             ,2
//                             ,21
//                             ,2
//                             ,2
//                             ,21
//                             ,2
//                             ,2
//                             ,21>;
void(* last_function)(std::vector<std::reference_wrapper<std::vector<int8_t, AlignedAllocator<int8_t, 64>>>>, std::vector<Box>&) = testlastfunction;
#endif