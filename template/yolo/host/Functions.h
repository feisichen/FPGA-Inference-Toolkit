#ifndef FUNCTION_H
#define FUNCTION_H
#include "AlignedAllocator.h"
#include <vector>
#include <deque>
#include <unordered_map>
#include <algorithm>
#include "Logger.h"
#include "Box.h"
#include <cmath>

#define scale_0_x 2.5111536979675293
#define scale_0_y 0.4974341094493866
#define scale_0_z 0.0039211893454194
#define zero_point_0_x -127
#define zero_point_0_y -128
#define zero_point_0_z -128
#define scale_1_x 2.4902994632720947
#define scale_1_y 1.8074067831039429
#define scale_1_z 0.0039214873686432
#define zero_point_1_x -128
#define zero_point_1_y -128
#define zero_point_1_z -128
#define scale_2_x 2.510971784591675
#define scale_2_y 3.0297296047210693
#define scale_2_z 0.0039213956333696
#define zero_point_2_x -126
#define zero_point_2_y -128
#define zero_point_2_z -128

#define CONFIDENCE_THRESHOLD 0.25 // 置信度阈值
#define PROB_THRESHOLD_LOW 0.3	  // 通道本身的置信度
#define PROB_THRESHOLD_HIGH 0.95
#define NMS_THRESHOLD 0.4 // NMS阈值

template<typename T>
auto rightShiftWithRound(T tmp, T shiftAmount) -> T
{
    T result = tmp >> shiftAmount;
    T result_less_shift = tmp >> (shiftAmount - 1);
    T carry = (result_less_shift & 1) > 0 ? 1 : 0;
    return (result + carry);
}

template<typename T, typename accum_T>
auto mul(T output, T output_sigmoid, T zeropoint_o, T zeropoint_i_sig, T zeropoint_o_mul, accum_T Num, accum_T N) -> T
{
    accum_T sum;
    sum = rightShiftWithRound((output - zeropoint_o) * (output_sigmoid - zeropoint_i_sig) * Num, N) + zeropoint_o_mul;
    output = (sum < -128 ? -128 : sum);
    return output;
}

bool sort_byConf(const Box &a, const Box &b)
{
    return a.conf > b.conf;
}

float calc_iou(Box *box1, Box *box2)
{
	float x1 = fmax(box1->x - box1->width / 2, box2->x - box2->width / 2);
	float y1 = fmax(box1->y - box1->height / 2, box2->y - box2->height / 2);
	float x2 = fmin(box1->x + box1->width / 2, box2->x + box2->width / 2);
	float y2 = fmin(box1->y + box1->height / 2, box2->y + box2->height / 2);

	float intersection = fmax(0, x2 - x1 + 1) * fmax(0, y2 - y1 + 1);
	float union_area = box1->width * box1->height + box2->width * box2->height - intersection;
	return intersection / union_area;
}
// template<typename T, int kernel_size, int threads_num>
// void csm(int layer_index, std::vector<std::vector<T,  AlignedAllocator<T, 64>>>& inputs, std::vector<std::vector<T,  AlignedAllocator<T, 64>>>& outputs){
//     std::vector<T,  AlignedAllocator<T, 64>>& input = inputs[0];
//     std::vector<T,  AlignedAllocator<T, 64>>& output = outputs[0];
//     std::vector<std::thread> threads;
//     for(int i=0; i < threads_num - 1; i++){
//         threads.push_back(std::thread(__csm_internal<T, kernel_size>, layer_index, threads_num, i, input, output));
//     }
//     __csm_internal<T, kernel_size>(layer_index, threads_num, threads_num - 1, input, output);
//     for(int i=0; i < threads_num - 1; i++){
//         threads[i].join();
//     }
//     return;
// }

template<typename T
        , typename accum_T
        , int rows
        , int cols
        , int channels
        , int filters
        , int kernel_size_x
        , int kernel_size_y
        , int pad_top
        , int pad_rt
        , int pad_bt
        , int pad_lt
        , int row_stride
        , int col_stride
        , const std::vector<T,  AlignedAllocator<T, 64>>& weights
        , T(*sigmoid)(T)
        , const std::vector<int>& bias_conv
        , const std::vector<int>& Num_conv
        , const std::vector<int>& N_conv
        , T zero_point_i
        , T zero_point_o
        , T zero_point_i_sig
        , T zero_point_o_mul
        , int Num_mul
        , int N_mul
>
void csm(int group_num
        , int group_index
        , std::vector<std::reference_wrapper<std::vector<T, AlignedAllocator<T, 64>>>> inputs
        , std::vector<std::reference_wrapper<std::vector<T, AlignedAllocator<T, 64>>>> outputs
)
{
    // LOG_DEBUG("csm -> group_num:%d, group_index:%d!", group_num, group_index);
    std::vector<T,  AlignedAllocator<T, 64>>& input = inputs[0].get();
    std::vector<T,  AlignedAllocator<T, 64>>& output = outputs[0].get();

    const int rows_out = (rows + pad_lt + pad_rt - kernel_size_x) / row_stride + 1;
    const int cols_out = (cols + pad_bt + pad_top - kernel_size_y) / col_stride + 1;
    const int rows_remain = (group_index < (rows_out % group_num)) ? 1 : 0;
    const int rows_per_group = rows_out / group_num;
    const int rows_todo = rows_per_group + rows_remain;
    const int rows_offset = (group_index * rows_per_group + std::min(group_index, rows_out % group_num));
    const int rows_begin = rows_offset * row_stride;
    const int rows_end = (rows_offset + rows_todo) * row_stride;
    const int cols_end = cols_out * col_stride;
    const int window_size = kernel_size_x * kernel_size_y;

    int output_index = rows_offset * cols_out * filters;

    for (int row_index = rows_begin; row_index < rows_end; row_index += row_stride)
    {
        for (int col_index = 0; col_index < cols_end; col_index += col_stride)
        {
            std::vector<T> pixel(window_size * channels, zero_point_i);
            for (int ch_index = 0; ch_index < channels; ++ch_index)
            {
                for (unsigned char k1 = 0; k1 < kernel_size_x; ++k1)
                {
                    if (k1 + row_index > pad_bt - 1 && k1 + row_index < rows + pad_bt)
                    {
                        for (unsigned char k2 = 0; k2 < kernel_size_y; ++k2)
                        {
                            if (k2 + col_index > pad_lt - 1 && k2 + col_index < cols + pad_lt)
                            {
                                // std::cout << input[((k1 + row_index - pad_bt) * cols + k2 + col_index - pad_lt) * channels + ch_index] << std::endl;
                                pixel[ch_index * window_size + k1 * kernel_size_y + k2] = input[((k1 + row_index - pad_bt) * cols + k2 + col_index - pad_lt) * channels + ch_index];
                            }
                        }
                    }
                }
            }
            int weight_index = 0;
            for (int filter_index = 0; filter_index < filters; ++filter_index)
            {
                accum_T sum = 0;
                for (int pixel_index = 0; pixel_index < window_size * channels; ++pixel_index, ++weight_index)
                {
                    sum += (pixel[pixel_index] - zero_point_i) * weights[weight_index];
                }
                sum = rightShiftWithRound<accum_T>((bias_conv[filter_index] + sum) * Num_conv[filter_index], N_conv[filter_index]);
                T temp = mul<T, accum_T>(sum + zero_point_o, sigmoid(sum), zero_point_o, zero_point_i_sig, zero_point_o_mul, Num_mul, N_mul);
                output[output_index++] = temp;
            }
        }
    }
}

template<typename T
        , typename accum_T
        , int rows
        , int cols
        , int channels
        , int filters
        , int kernel_size_x
        , int kernel_size_y
        , int pad_top
        , int pad_rt
        , int pad_bt
        , int pad_lt
        , int row_stride
        , int col_stride
        , const std::vector<T,  AlignedAllocator<T, 64>>& weights
        , const std::vector<int>& bias_conv
        , const std::vector<int>& Num_conv
        , const std::vector<int>& N_conv
        , T zero_point_i
        , T zero_point_o
>
void csm_last(int group_num
        , int group_index
        , std::vector<std::reference_wrapper<std::vector<T, AlignedAllocator<T, 64>>>> inputs
        , std::vector<std::reference_wrapper<std::vector<T, AlignedAllocator<T, 64>>>> outputs
)
{
    // LOG_DEBUG("csm -> group_num:%d, group_index:%d!", group_num, group_index);
    std::vector<T,  AlignedAllocator<T, 64>>& input = inputs[0].get();
    std::vector<T,  AlignedAllocator<T, 64>>& output = outputs[0].get();

    const int rows_out = (rows + pad_lt + pad_rt - kernel_size_x) / row_stride + 1;
    const int cols_out = (cols + pad_bt + pad_top - kernel_size_y) / col_stride + 1;
    const int rows_remain = (group_index < (rows_out % group_num)) ? 1 : 0;
    const int rows_per_group = rows_out / group_num;
    const int rows_todo = rows_per_group + rows_remain;
    const int rows_offset = (group_index * rows_per_group + std::min(group_index, rows_out % group_num));
    const int rows_begin = rows_offset * row_stride;
    const int rows_end = (rows_offset + rows_todo) * row_stride;
    const int cols_end = cols_out * col_stride;
    const int window_size = kernel_size_x * kernel_size_y;

    int output_index = rows_offset * cols_out * filters;

    for (int row_index = rows_begin; row_index < rows_end; row_index += row_stride)
    {
        for (int col_index = 0; col_index < cols_end; col_index += col_stride)
        {
            std::vector<T> pixel(window_size * channels, zero_point_i);
            for (int ch_index = 0; ch_index < channels; ++ch_index)
            {
                for (unsigned char k1 = 0; k1 < kernel_size_x; ++k1)
                {
                    if (k1 + row_index > pad_bt - 1 && k1 + row_index < rows + pad_bt)
                    {
                        for (unsigned char k2 = 0; k2 < kernel_size_y; ++k2)
                        {
                            if (k2 + col_index > pad_lt - 1 && k2 + col_index < cols + pad_lt)
                            {
                                // std::cout << input[((k1 + row_index - pad_bt) * cols + k2 + col_index - pad_lt) * channels + ch_index] << std::endl;
                                pixel[ch_index * window_size + k1 * kernel_size_y + k2] = input[((k1 + row_index - pad_bt) * cols + k2 + col_index - pad_lt) * channels + ch_index];
                            }
                        }
                    }
                }
            }
            int weight_index = 0;
            for (int filter_index = 0; filter_index < filters; ++filter_index)
            {
                accum_T sum = 0;
                for (int pixel_index = 0; pixel_index < window_size * channels; ++pixel_index, ++weight_index)
                {
                    sum += (pixel[pixel_index] - zero_point_i) * weights[weight_index];
                }
                sum = rightShiftWithRound<accum_T>((bias_conv[filter_index] + sum) * Num_conv[filter_index], N_conv[filter_index]);
                output[output_index++] = sum + zero_point_o;
            }
        }
    }
}

template<typename T
        , int num_filters_x
        , int num_filters_y
        , int height
        , int width
        , int Num_x
        , int N_x
        , int Num_y
        , int N_y
        , T zero_point_ix
        , T zero_point_iy
        , T zero_point_o
>
void concat(int group_num
        , int group_index
        , std::vector<std::reference_wrapper<std::vector<T, AlignedAllocator<T, 64>>>> inputs
        , std::vector<std::reference_wrapper<std::vector<T, AlignedAllocator<T, 64>>>> outputs
)
{
    std::vector<T,  AlignedAllocator<T, 64>>& input_y = inputs[0].get();
    std::vector<T,  AlignedAllocator<T, 64>>& input_x = inputs[1].get();
    std::vector<T,  AlignedAllocator<T, 64>>& output = outputs[0].get();

    const int height_remain = (group_index < (height % group_num)) ? 1 : 0;
    const int height_per_group = height / group_num;
    const int height_todo = height_per_group + height_remain;
    const int height_begin = (group_index * height_per_group + std::min(group_index, height % group_num));
    const int height_end = height_begin + height_todo;

    int tmp, sum;
    T pixel;

    int indexs_x = height_begin * width * num_filters_x;
    int indexs_y = height_begin * width * num_filters_y;
    int output_index = indexs_x + indexs_y;

    for (int i = height_begin;i < height_end; ++i) {
        for (int j = 0; j < width; ++j) {
            for (int k = 0; k < num_filters_y; ++k){
                pixel = input_y[indexs_y++];
                tmp = (pixel - zero_point_iy) * Num_y;
                sum = (tmp >> N_y) + ((tmp >> (N_y - 1)) & 1) + zero_point_o;
                output[output_index++] = sum < -128 ? -128 : sum;
            }
            for (int k = 0; k < num_filters_x; ++k){
                pixel = input_x[indexs_x++];
                tmp = (pixel - zero_point_ix) * Num_x;
                sum = (tmp >> N_x) + ((tmp >> (N_x - 1)) & 1) + zero_point_o;
                output[output_index++] = sum < -128 ? -128 : sum;
            }
        }
    }
}

template<typename T
        , int num_filters_x
        , int num_filters_y
        , int height
        , int width
>
void concat_without_quant(int group_num
        , int group_index
        , std::vector<std::reference_wrapper<std::vector<T, AlignedAllocator<T, 64>>>> inputs
        , std::vector<std::reference_wrapper<std::vector<T, AlignedAllocator<T, 64>>>> outputs
)
{
    std::vector<T,  AlignedAllocator<T, 64>>& input_y = inputs[0].get();
    std::vector<T,  AlignedAllocator<T, 64>>& input_x = inputs[1].get();
    std::vector<T,  AlignedAllocator<T, 64>>& output = outputs[0].get();

    const int height_remain = (group_index < (height % group_num)) ? 1 : 0;
    const int height_per_group = height / group_num;
    const int height_todo = height_per_group + height_remain;
    const int height_begin = (group_index * height_per_group + std::min(group_index, height % group_num));
    const int height_end = height_begin + height_todo;

    int indexs_x = height_begin * width * num_filters_x;
    int indexs_y = height_begin * width * num_filters_y;
    int output_index = indexs_x + indexs_y;

    for (int i = height_begin;i < height_end; ++i) {
        for (int j = 0; j < width; ++j) {
            for (int k = 0; k < num_filters_y; ++k){
                output[output_index++] = input_y[indexs_y++];
            }
            for (int k = 0; k < num_filters_x; ++k){
                output[output_index++] = input_x[indexs_x++];
            }
        }
    }
}

// template<typename T
//         , const std::vector<int>& num_filters
//         , int height
//         , int width
//         , const std::vector<int>& Num_concat
//         , const std::vector<int>& N_concat
//         , const std::vector<int>& zero_point_i_concat
//         , T zero_point_o_concat
// >
// void concat_float(int group_num
//         , int group_index
//         , std::vector<std::reference_wrapper<std::vector<T, AlignedAllocator<T, 64>>>> inputs
//         , std::vector<std::vector<float>>& outputs
// )
// {
//     std::vector<T,  AlignedAllocator<T, 64>>& output = outputs[0].get();

//     const int height_remain = (group_index < (height % group_num)) ? 1 : 0;
//     const int height_per_group = height / group_num;
//     const int height_todo = height_per_group + height_remain;
//     const int height_begin = (group_index * height_per_group + std::min(group_index, height % group_num));
//     const int height_end = height_begin + height_todo;

//     int tmp, sum;
//     T pixel;

//     int inputs_size = inputs.size();
//     std::vector<int> indexs(inputs_size);
//     int output_index = 0;
//     for(int i=0; i<inputs_size;i++){
//         indexs[i] = height_begin * width * num_filters[i];
//         output_index += height_begin * width * num_filters[i];
//     }

//     for (int i = height_begin;i < height_end; ++i) {
//         for (int j = 0; j < width; ++j) {
//             for(int t = 0; t < inputs_size; t++){
//                 for (int k = 0; k < num_filters[t]; ++k){
//                     pixel = inputs[t].get()[indexs[t]++];
//                     tmp = (pixel - zero_point_i_concat[t]) * Num_concat[t];
//                     sum = (tmp >> N_concat[t]) + ((tmp >> (N_concat[t] - 1)) & 1) + zero_point_o_concat;
//                     output[output_index++] = sum < -128 ? -128 : sum;
//                 }
//             }
//         }
//     }
// }

template<typename T
        , int num_filters
        , int height
        , int width
        , int Num_add_x
        , int N_add_x
        , int Num_add_y
        , int N_add_y
        , T zero_point_ix_add
        , T zero_point_iy_add
        , T zero_point_o_add
>
void add(int group_num
        , int group_index
        , std::vector<std::reference_wrapper<std::vector<T, AlignedAllocator<T, 64>>>> inputs
        , std::vector<std::reference_wrapper<std::vector<T, AlignedAllocator<T, 64>>>> outputs
)
{
    std::vector<T,  AlignedAllocator<T, 64>>& input_x = inputs[0].get();
    std::vector<T,  AlignedAllocator<T, 64>>& input_y = inputs[1].get();
    std::vector<T,  AlignedAllocator<T, 64>>& output = outputs[0].get();

    const int height_remain = (group_index < (height % group_num)) ? 1 : 0;
    const int height_per_group = height / group_num;
    const int height_todo = height_per_group + height_remain;
    const int height_begin = (group_index * height_per_group + std::min(group_index, height % group_num));
    const int height_end = height_begin + height_todo;

    // std::cout << input_x.size() << std::endl;
    // std::cout << input_y.size() << std::endl;
    // std::cout << num_filters * height * width << std::endl;
    int x, y, sum;

    int x_index = height_begin * width * num_filters;
    int y_index = height_begin * width * num_filters;
    int output_index = height_begin * width * num_filters;

    for (int i = height_begin;i < height_end; ++i) {
        for (int j = 0; j < width; ++j) {
            for (int k = 0; k < num_filters; ++k){
                x = input_x[x_index++] - zero_point_ix_add;
                y = input_y[y_index++] - zero_point_iy_add;
                sum = rightShiftWithRound<int>(x * Num_add_x + y * Num_add_y, N_add_x) + zero_point_o_add;
                output[output_index++] = sum < -128 ? -128 : sum;
            }
        }
    }
}

template<typename T
        , int num_filters
        , int height
        , int width
        , int Num_mul
        , int N_mul
        , T zero_point_ix_mul
        , T zero_point_iy_mul
        , T zero_point_o_mul
>
void mul(int group_num
        , int group_index
        , std::vector<std::reference_wrapper<std::vector<T, AlignedAllocator<T, 64>>>> inputs
        , std::vector<std::reference_wrapper<std::vector<T, AlignedAllocator<T, 64>>>> outputs
)
{
    std::vector<T,  AlignedAllocator<T, 64>>& input_y = inputs[0].get();
    std::vector<T,  AlignedAllocator<T, 64>>& input_x = inputs[1].get();
    std::vector<T,  AlignedAllocator<T, 64>>& output = outputs[0].get();

    const int height_remain = (group_index < (height % group_num)) ? 1 : 0;
    const int height_per_group = height / group_num;
    const int height_todo = height_per_group + height_remain;
    const int height_begin = (group_index * height_per_group + std::min(group_index, height % group_num));
    const int height_end = height_begin + height_todo;

    int x, y, sum;

    int x_index = height_begin * width * num_filters;
    int y_index = height_begin * width * num_filters;
    int output_index = height_begin * width * num_filters;

    for (int i = height_begin;i < height_end; ++i) {
        for (int j = 0; j < width; ++j) {
            for (int k = 0; k < num_filters; ++k){
                y = input_y[x_index++] - zero_point_iy_mul;
                x = input_x[y_index++] - zero_point_ix_mul;
                sum = rightShiftWithRound<int>(x * y * Num_mul, N_mul) + zero_point_o_mul;
                output[output_index++] = sum < -128 ? -128 : sum;
            }
        }
    }
}

template<typename T
        , T m_value
        , int num_filters
        , int height
        , int width
        , int Num_mul
        , int N_mul
        , T zero_point_ix_mul
        , T zero_point_iy_mul
        , T zero_point_o_mul
>
void mul_const(int group_num
        , int group_index
        , std::vector<std::reference_wrapper<std::vector<T, AlignedAllocator<T, 64>>>> inputs
        , std::vector<std::reference_wrapper<std::vector<T, AlignedAllocator<T, 64>>>> outputs
)
{
    std::vector<T,  AlignedAllocator<T, 64>>& input_y = inputs[0].get();
    std::vector<T,  AlignedAllocator<T, 64>>& output = outputs[0].get();

    const int height_remain = (group_index < (height % group_num)) ? 1 : 0;
    const int height_per_group = height / group_num;
    const int height_todo = height_per_group + height_remain;
    const int height_begin = (group_index * height_per_group + std::min(group_index, height % group_num));
    const int height_end = height_begin + height_todo;

    int x, y, sum;

    int x_index = height_begin * width * num_filters;
    int output_index = height_begin * width * num_filters;

    for (int i = height_begin;i < height_end; ++i) {
        for (int j = 0; j < width; ++j) {
            for (int k = 0; k < num_filters; ++k){
                y = input_y[x_index++] - zero_point_iy_mul;
                x = m_value - zero_point_ix_mul;
                sum = rightShiftWithRound<int>(x * y * Num_mul, N_mul) + zero_point_o_mul;
                output[output_index++] = sum < -128 ? -128 : sum;
            }
        }
    }
}

template<typename T
        , int num_filters
        , int height
        , int width
        , T(*pow_split)(T)
>
void pow(int group_num
        , int group_index
        , std::vector<std::reference_wrapper<std::vector<T, AlignedAllocator<T, 64>>>> inputs
        , std::vector<std::reference_wrapper<std::vector<T, AlignedAllocator<T, 64>>>> outputs
)
{
    std::vector<T,  AlignedAllocator<T, 64>>& input = inputs[0].get();
    std::vector<T,  AlignedAllocator<T, 64>>& output = outputs[0].get();

    const int height_remain = (group_index < (height % group_num)) ? 1 : 0;
    const int height_per_group = height / group_num;
    const int height_todo = height_per_group + height_remain;
    const int height_begin = (group_index * height_per_group + std::min(group_index, height % group_num));
    const int height_end = height_begin + height_todo;

    T x, sum;

    int x_index = height_begin * width * num_filters;
    int output_index = height_begin * width * num_filters;

    for (int i = height_begin;i < height_end; ++i) {
        for (int j = 0; j < width; ++j) {
            for (int k = 0; k < num_filters; ++k){
                x = input[x_index++];
                sum = pow_split(x);
                output[output_index++] = sum;
            }
        }
    }
}

template<typename T
        , int height
        , int width
        , int channel
        , T(*sigmoid)(T)
        , T zero_point_o
        , int c1
>
void transpose(int group_num
        , int group_index
        , std::vector<std::reference_wrapper<std::vector<T, AlignedAllocator<T, 64>>>> inputs
        , std::vector<std::reference_wrapper<std::vector<T, AlignedAllocator<T, 64>>>> outputs
)
{
    // std::cout << inputs.size() <<std::endl;
    // std::cout << outputs.size() <<std::endl;
    std::vector<T,  AlignedAllocator<T, 64>>& input = inputs[0].get();
    std::vector<T,  AlignedAllocator<T, 64>>& output = outputs[0].get();
    
    const int height_remain = (group_index < (height % group_num)) ? 1 : 0;
    const int height_per_group = height / group_num;
    const int height_todo = height_per_group + height_remain;
    const int height_begin = (group_index * height_per_group + std::min(group_index, height % group_num));
    const int height_end = height_begin + height_todo;

    const int c2 = channel / c1;

	for (int c = 0; c < c1; c++)
	{
		for (int h = height_begin; h < height_end; h++)
		{
			for (int w = 0; w < width; w++)
			{
				for (int d = 0; d < c2; d++)
				{
					output[c * height * width * c2 + h * width * c2 + w * c2 + d] = sigmoid(input[h * width * c1 * c2 + w * c1 * c2 + c * c2 + d] - zero_point_o);
				}
			}
		}
	}
}

template<typename T
        , int height_in
        , int width_in
        , int channel
>
void resize(int group_num
        , int group_index
        , std::vector<std::reference_wrapper<std::vector<T, AlignedAllocator<T, 64>>>> inputs
        , std::vector<std::reference_wrapper<std::vector<T, AlignedAllocator<T, 64>>>> outputs
)
{
    std::vector<T,  AlignedAllocator<T, 64>>& input = inputs[0].get();
    std::vector<T,  AlignedAllocator<T, 64>>& output = outputs[0].get();

	int height = height_in * 2;
	int width = width_in * 2;

    const int height_remain = (group_index < (height % group_num)) ? 1 : 0;
    const int height_per_group = height / group_num;
    const int height_todo = height_per_group + height_remain;
    const int height_begin = (group_index * height_per_group + std::min(group_index, height % group_num));
    const int height_end = height_begin + height_todo;

	for (int i = height_begin; i < height_end; i++)
	{
		for (int j = 0; j < width; j++)
		{
			for (int k = 0; k < channel; k++)
			{
				output[j * channel + k + i * width * channel] = input[j / 2 * channel + k + i / 2 * width / 2 * channel];
			}
		}
	}
}

template<typename T
        , int height
        , int width
        , int size1
        , int size2
        , int size3
>
void splitx3(int group_num
        , int group_index
        , std::vector<std::reference_wrapper<std::vector<T, AlignedAllocator<T, 64>>>> inputs
        , std::vector<std::reference_wrapper<std::vector<T, AlignedAllocator<T, 64>>>> outputs
)
{
    std::vector<T,  AlignedAllocator<T, 64>>& input = inputs[0].get();
    std::vector<T,  AlignedAllocator<T, 64>>& output1 = outputs[0].get();
    std::vector<T,  AlignedAllocator<T, 64>>& output2 = outputs[1].get();
    std::vector<T,  AlignedAllocator<T, 64>>& output3 = outputs[2].get();

    const int height_remain = (group_index < (height % group_num)) ? 1 : 0;
    const int height_per_group = height / group_num;
    const int height_todo = height_per_group + height_remain;
    const int height_begin = (group_index * height_per_group + std::min(group_index, height % group_num));
    const int height_end = height_begin + height_todo;

	for (int c = 0; c < 3; c++)
	{
		for (int h = height_begin; h < height_end; h++)
		{
			for (int w = 0; w < width; w++)
			{
				for (int d = 0; d < size1; d++)
				{
					output1[c * height * width * size1 + h * width * size1 + w * size1 + d] = input[c * height * width * 25 + h * width * 25 + w * 25 + d];
                }
                for (int d = 0; d < size2; d++)
				{
					output2[c * height * width * size2 + h * width * size2 + w * size2 + d] = input[c * height * width * 25 + h * width * 25 + w * 25 + d + size1];                
                }
                for (int d = 0; d < size3; d++)
				{
					output3[c * height * width * size3 + h * width * size3 + w * size3 + d] = input[c * height * width * 25 + h * width * 25 + w * 25 + d + size1 + size2];
                }
			}
		}
	}
}

template<typename T
        , int rows
        , int cols
        , int channels
        , int kernel_size_x
        , int kernel_size_y
        , int pad_top
        , int pad_rt
        , int pad_bt
        , int pad_lt
        , int row_stride
        , int col_stride
        , T zero_point_i
>
void maxpool(int group_num
        , int group_index
        , std::vector<std::reference_wrapper<std::vector<T, AlignedAllocator<T, 64>>>> inputs
        , std::vector<std::reference_wrapper<std::vector<T, AlignedAllocator<T, 64>>>> outputs
)
{
    struct node {
        int idx;
        T value;
        node() {}
        node(int idx_, T v) :idx(idx_), value(v) {}
    };

    std::vector<T, AlignedAllocator<T, 64>>& input = inputs[0].get();
    std::vector<T, AlignedAllocator<T, 64>>& output = outputs[0].get();

    const int rows_out = (rows + pad_lt + pad_rt - kernel_size_x) / row_stride + 1;
    const int cols_out = (cols + pad_bt + pad_top - kernel_size_y) / col_stride + 1;
    const int rows_remain = (group_index < (rows_out % group_num)) ? 1 : 0;
    const int rows_per_group = rows_out / group_num;
    const int rows_todo = rows_per_group + rows_remain;
    const int rows_offset = (group_index * rows_per_group + std::min(group_index, rows_out % group_num));
    const int rows_begin = rows_offset * row_stride;
    const int rows_end = (rows_offset + rows_todo - 1) * row_stride + kernel_size_x;
    const int cols_end = cols + pad_bt + pad_top;

    int output_index = rows_offset * cols_out * channels;

    std::vector<std::vector<T>> cmax(channels, std::vector<T>(rows_todo* cols_end));

    int cmax_col_idx = 0;
    for (int col_index = 0 ; col_index < cols_end; ++col_index, ++cmax_col_idx)
    {
        std::vector<std::deque<node>> q(channels, std::deque<node>());
        int cmax_row_idx = 0;
        if (col_index > pad_lt - 1 && col_index < cols + pad_lt)
        {
            for (unsigned char k1 = 0; k1 < kernel_size_x; ++k1)
            {
                if (rows_begin + k1 > pad_bt - 1 && rows_begin + k1 < rows + pad_bt)
                {
                    for (int ch_index = 0; ch_index < channels; ++ch_index)
                    {
                        T pixel = input[((rows_begin + k1 - pad_bt) * cols + col_index - pad_lt) * channels + ch_index];
                        while (!q[ch_index].empty() && pixel >= q[ch_index].back().value)
                        {
                            q[ch_index].pop_back();
                        }
                        q[ch_index].emplace_back(rows_begin + k1, pixel);
                    }
                }
                else {
                    for (int ch_index = 0; ch_index < channels; ++ch_index)
                    {
                        T pixel = zero_point_i;
                        while (!q[ch_index].empty() && pixel >= q[ch_index].back().value)
                        {
                            q[ch_index].pop_back();
                        }
                        q[ch_index].emplace_back(rows_begin + k1, pixel);
                    }
                }
            }
        }
        else {
            for (int ch_index = 0; ch_index < channels; ++ch_index) {
                T pixel = zero_point_i;
                while (!q[ch_index].empty() && pixel >= q[ch_index].back().value) {
                    q[ch_index].pop_back();
                }
                q[ch_index].emplace_back(kernel_size_x - 1, pixel);
            }
        }
        for (int ch_index = 0; ch_index < channels; ++ch_index)
        {
            cmax[ch_index][(cmax_row_idx) * cols_end + cmax_col_idx] = q[ch_index].front().value;
        }
        ++cmax_row_idx;
        for (int row_index = rows_begin + kernel_size_x; row_index < rows_end; row_index += row_stride, ++cmax_row_idx)
        {
            if (col_index > pad_lt - 1 && col_index < cols + pad_lt)
            {
                for (unsigned char k1 = 0; k1 < row_stride; ++k1)
                {
                    if (row_index + k1 > pad_bt - 1 && row_index + k1 < rows + pad_bt)
                    {
                        for (int ch_index = 0; ch_index < channels; ++ch_index)
                        {
                            T pixel = input[((row_index + k1 - pad_bt) * cols + col_index - pad_lt) * channels + ch_index];
                            while (!q[ch_index].empty() && pixel >= q[ch_index].back().value)
                            {
                                q[ch_index].pop_back();
                            }
                            q[ch_index].emplace_back(row_index + k1, pixel);
                        }
                    }
                    else {
                        for (int ch_index = 0; ch_index < channels; ++ch_index)
                        {
                            T pixel = zero_point_i;
                            while (!q[ch_index].empty() && pixel >= q[ch_index].back().value)
                            {
                                q[ch_index].pop_back();
                            }
                            q[ch_index].emplace_back(row_index + k1, pixel);
                        }
                    }
                }
            }
            else {
                for (int ch_index = 0; ch_index < channels; ++ch_index) {
                    T pixel = zero_point_i;
                    while (!q[ch_index].empty() && pixel >= q[ch_index].back().value) {
                        q[ch_index].pop_back();
                    }
                    q[ch_index].emplace_back(row_index + row_stride - 1, pixel);
                }
            }
            for (int ch_index = 0; ch_index < channels; ++ch_index)
            {
                while (q[ch_index].front().idx < row_index + row_stride - kernel_size_x) {
                    q[ch_index].pop_front();
                }
                cmax[ch_index][cmax_row_idx * cols_end + cmax_col_idx] = q[ch_index].front().value;
            }
        }
    }

    for (int row_index = 0; row_index < rows_todo; ++row_index)
    {
        std::vector<std::deque<node>> q(channels, std::deque<node>());
        for (unsigned char k2 = 0; k2 < kernel_size_y; ++k2)
        {
            for (int ch_index = 0; ch_index < channels; ++ch_index)
            {
                T pixel = cmax[ch_index][row_index * cols_end + k2];
                while (!q[ch_index].empty() && pixel >= q[ch_index].back().value)
                {
                    q[ch_index].pop_back();
                }
                q[ch_index].emplace_back(k2, pixel);
            }
        }
        for (int ch_index = 0; ch_index < channels; ++ch_index)
        {
            output[output_index++] = q[ch_index].front().value;
        }
        for (int col_index = kernel_size_y; col_index < cols_end; col_index += col_stride)
        {
            for (unsigned char k2 = 0; k2 < col_stride; ++k2)
            {
                for (int ch_index = 0; ch_index < channels; ++ch_index)
                {
                    T pixel = cmax[ch_index][row_index * cols_end + col_index + k2];
                    while (!q[ch_index].empty() && pixel >= q[ch_index].back().value)
                    {
                        q[ch_index].pop_back();
                    }
                    q[ch_index].emplace_back(col_index + k2, pixel);
                }
            }
            for (int ch_index = 0; ch_index < channels; ++ch_index)
            {
                while (q[ch_index].front().idx < col_index + col_stride - kernel_size_y) {
                    q[ch_index].pop_front();
                }
                output[output_index++] = q[ch_index].front().value;
            }
        }
    }
}

template<typename T
        , int box_num
        , int height_0
        , int height_1
        , int height_2
        , int width_0
        , int width_1
        , int width_2
        , int channel_0_0
        , int channel_0_1
        , int channel_0_2
        , int channel_1_0
        , int channel_1_1
        , int channel_1_2
        , int channel_2_0
        , int channel_2_1
        , int channel_2_2
>
void detect(std::vector<std::reference_wrapper<std::vector<T, AlignedAllocator<T, 64>>>> inputs
        , std::vector<Box>& output
)
{
    int channel_0 = channel_0_0 + channel_0_1 + channel_0_2;
    int size_0 = 3 * height_0 * width_0 * (channel_0);
    std::vector<float> concat_0(size_0);  
    std::vector<T, AlignedAllocator<T, 64>>& split_0_x = inputs[0].get();
    std::vector<T, AlignedAllocator<T, 64>>& split_0_y = inputs[1].get();
    std::vector<T, AlignedAllocator<T, 64>>& split_0_z = inputs[2].get();
    for (int c = 0; c < 3; c++)
    {
        for (int h = 0; h < height_0; h++)
        {
            for (int w = 0; w < width_0; w++)
            {
                for (int d = 0, t = 0; d < channel_0_0; d++, t++)
                {
                    concat_0[c * height_0 * width_0 * channel_0 + h * width_0 * channel_0 + w * channel_0 + t] = float(split_0_x[c * height_0 * width_0 * channel_0_0 + h * width_0 * channel_0_0 + w * channel_0_0 + d] - zero_point_0_x) * scale_0_x;
                }
                for (int d = 0, t = channel_0_0; d < channel_0_1; d++, t++)
                {
                    concat_0[c * height_0 * width_0 * channel_0 + h * width_0 * channel_0 + w * channel_0 + t] = float(split_0_y[c * height_0 * width_0 * channel_0_1 + h * width_0 * channel_0_1 + w * channel_0_1 + d] - zero_point_0_y) * scale_0_y;
                }
                for (int d = 0, t = channel_0_0 + channel_0_1; d < channel_0_2; d++, t++)
                {
                    concat_0[c * height_0 * width_0 * channel_0 + h * width_0 * channel_0 + w * channel_0 + t] = float(split_0_z[c * height_0 * width_0 * channel_0_2 + h * width_0 * channel_0_2 + w * channel_0_2 + d] - zero_point_0_z) * scale_0_z;
                }
            }
        }
    }
    int channel_1 = channel_1_0 + channel_1_1 + channel_1_2;
    int size_1 = 3 * height_1 * width_1 * (channel_1);
    std::vector<float> concat_1(size_1);  
    std::vector<T, AlignedAllocator<T, 64>>& split_1_x = inputs[3].get();
    std::vector<T, AlignedAllocator<T, 64>>& split_1_y = inputs[4].get();
    std::vector<T, AlignedAllocator<T, 64>>& split_1_z = inputs[5].get();
    for (int c = 0; c < 3; c++)
    {
        for (int h = 0; h < height_1; h++)
        {
            for (int w = 0; w < width_1; w++)
            {
                for (int d = 0, t = 0; d < channel_1_0; d++, t++)
                {
                    concat_1[c * height_1 * width_1 * channel_1 + h * width_1 * channel_1 + w * channel_1 + t] = float(split_1_x[c * height_1 * width_1 * channel_1_0 + h * width_1 * channel_1_0 + w * channel_1_0 + d] - zero_point_1_x) * scale_1_x;
                }
                for (int d = 0, t = channel_1_0; d < channel_1_1; d++, t++)
                {
                    concat_1[c * height_1 * width_1 * channel_1 + h * width_1 * channel_1 + w * channel_1 + t] = float(split_1_y[c * height_1 * width_1 * channel_1_1 + h * width_1 * channel_1_1 + w * channel_1_1 + d] - zero_point_1_y) * scale_1_y;
                }
                for (int d = 0, t = channel_1_0 + channel_1_1; d < channel_1_2; d++, t++)
                {
                    concat_1[c * height_1 * width_1 * channel_1 + h * width_1 * channel_1 + w * channel_1 + t] = float(split_1_z[c * height_1 * width_1 * channel_1_2 + h * width_1 * channel_1_2 + w * channel_1_2 + d] - zero_point_1_z) * scale_1_z;
                }
            }
        }
    }
    int channel_2 = channel_2_0 + channel_2_1 + channel_2_2;
    int size_2 = 3 * height_2 * width_2 * (channel_2);
    std::vector<float> concat_2(size_2);  
    std::vector<T, AlignedAllocator<T, 64>>& split_2_x = inputs[6].get();
    std::vector<T, AlignedAllocator<T, 64>>& split_2_y = inputs[7].get();
    std::vector<T, AlignedAllocator<T, 64>>& split_2_z = inputs[8].get();
    for (int c = 0; c < 3; c++)
    {
        for (int h = 0; h < height_2; h++)
        {
            for (int w = 0; w < width_2; w++)
            {
                for (int d = 0, t = 0; d < channel_2_0; d++, t++)
                {
                    concat_2[c * height_2 * width_2 * channel_2 + h * width_2 * channel_2 + w * channel_2 + t] = float(split_2_x[c * height_2 * width_2 * channel_2_0 + h * width_2 * channel_2_0 + w * channel_2_0 + d] - zero_point_2_x) * scale_2_x;
                }
                for (int d = 0, t = channel_2_0; d < channel_2_1; d++, t++)
                {
                    concat_2[c * height_2 * width_2 * channel_2 + h * width_2 * channel_2 + w * channel_2 + t] = float(split_2_y[c * height_2 * width_2 * channel_2_1 + h * width_2 * channel_2_1 + w * channel_2_1 + d] - zero_point_2_y) * scale_2_y;
                }
                for (int d = 0, t = channel_2_0 + channel_2_1; d < channel_2_2; d++, t++)
                {
                    concat_2[c * height_2 * width_2 * channel_2 + h * width_2 * channel_2 + w * channel_2 + t] = float(split_2_z[c * height_2 * width_2 * channel_2_2 + h * width_2 * channel_2_2 + w * channel_2_2 + d] - zero_point_2_z) * scale_2_z;
                }
            }
        }
    }

    std::vector<float> inputs_concat(size_0 + size_1 + size_2);
	for (int i = 0; i < box_num; i++)
	{
		for (int j = 0; j < 25; j++)
		{
			if (i < size_0 / 25)
				inputs_concat[i * 25 + j] = concat_0[i * 25 + j];
			else if (i < size_0 / 25 + size_1 / 25)
				inputs_concat[i * 25 + j] = concat_1[(i - size_0 / 25) * 25 + j];
			else
				inputs_concat[i * 25 + j] = concat_2[(i - size_0 / 25 - size_1 / 25) * 25 + j];
		}
	}
    // for(auto e:inputs_concat){
    //     std::cout << float(e)  << std::endl;
    // }
    concat_0.clear();
    concat_1.clear();
    concat_2.clear();
	std::vector<Box> boxes(box_num);
	for (int i = 0; i < box_num; i++)
	{
		float max = 0;
		int classid = 0;
		for (int j = 5; j < channel_0; j++)
		{
			if (inputs_concat[i * channel_0 + j] > max)
			{
				max = inputs_concat[i * channel_0 + j];
				// 需减去前面4个不是类别的元素，以及背景类，但考虑到数组序号从0开始，未减去
				classid = j - 4;
			}
		}
		boxes[i].class_id = classid;
		if (inputs_concat[i * channel_0 + 4] < PROB_THRESHOLD_HIGH && inputs_concat[i * channel_0 + 4] > PROB_THRESHOLD_LOW)
			boxes[i].conf = max * inputs_concat[i * channel_0 + 4];
		else
			boxes[i].conf = 0;
		boxes[i].x = inputs_concat[i * channel_0];
		boxes[i].y = inputs_concat[i * channel_0 + 1];
		boxes[i].width = inputs_concat[i * channel_0 + 2];
		boxes[i].height = inputs_concat[i * channel_0 + 3];
	}
    // std::cout << "init" << std::endl;
    int i, j;
	// 根据置信度排序
	std::sort(boxes.begin(), boxes.end(), sort_byConf);
	for (i = 0; i < box_num; i++)
	{
		if (boxes[i].conf >= CONFIDENCE_THRESHOLD)
		{
			for (j = i + 1; j < box_num; j++)
			{
                // if(boxes[j].class_id == 6 && boxes[j].conf > 0.71){
                //     printf("%d | %f | %f | %f | %f | %f\n", boxes[j].class_id, boxes[j].x, boxes[j].y, boxes[j].width, boxes[j].height, boxes[j].conf);
                // }
				if (boxes[j].class_id == boxes[i].class_id && calc_iou(&boxes[i], &boxes[j]) > NMS_THRESHOLD)
				{

					boxes[j].conf = 0; // 将置信度设为0表示此框被抑制
				}
			}
		}
	}
    // std::cout << "nms over" << std::endl;
    for (int i = 0; i < 25200; i++)
	{
		if (boxes[i].conf > CONFIDENCE_THRESHOLD)
		{
			output.push_back(boxes[i]);
		}
	}
}

#endif