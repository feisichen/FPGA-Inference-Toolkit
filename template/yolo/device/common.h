#ifndef COMMON_H_
#define COMMON_H_
#include "type.h"
// macro for integer division
#define DIVIDE(a, b) ((const unsigned int)((a) / (b)))

// macro for evaluating min and max
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

int rightShiftWithRound(int tmp, int shiftAmount)
{
	int result = tmp >> shiftAmount;
	int result_less_shift = tmp >> (shiftAmount - 1);
	int carry = (result_less_shift & 1) > 0 ? 1 : 0;
	return (result + carry);
}

int8_t mul(int8_t output, int8_t output_sigmoid, int8_t zeropoint_o, int8_t zeropoint_i_sig, int8_t zeropoint_o_mul, int Num, int N)
{
	int sum = 0;
	sum = rightShiftWithRound((output - zeropoint_o) * (output_sigmoid - zeropoint_i_sig) * Num, N) + zeropoint_o_mul;
	output = (sum < -128 ? -128 : sum);
	return output;
}

#define WINDOW_CHANNEL_DEPTH 1

#endif