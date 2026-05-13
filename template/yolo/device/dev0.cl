#include "type.h"
#include "common.h"
#include "concat.h"
#include "mem_read.h"
#include "mem_write.h"
#include "add.h"
#include "weights.h"
#include "csm.h"
#include "fork.h"
#pragma OPENCL EXTENSION cl_intel_channels : enable

//Layer 0
#define COARSE_LAYER_0 8

channel int8_t csm_in_0[1] __attribute__((depth(3)));
channel int8_t csm_out_0[2] __attribute__((depth(OUTPUT_CHANNELS_LAYER_0 * OUTPUT_SIZE_LAYER_0)));

MEM_READ(1, 0, csm_in_0, 640 * 640 * 3)
// GEN_CSM_LAYER(0, csm_in_0, csm_out_0, COARSE_LAYER_0, int8_t, int)
GEN_FINE_CSM_LAYER(1, 2, 0, csm_in_0, csm_out_0, COARSE_LAYER_0, int8_t, int)
//Layer 1
#define COARSE_LAYER_1 8

channel int8_t glue_out_1[1] __attribute__((depth(OUTPUT_CHANNELS_LAYER_1)));
// channel int8_t csm_out_1 __attribute__((depth(OUTPUT_CHANNELS_LAYER_1)));
// channel int8_t csm_in_5 __attribute__((depth(OUTPUT_CHANNELS_LAYER_1)));

GEN_CSM_LAYER(2, 1, 1, csm_out_0, glue_out_1, COARSE_LAYER_1, int8_t, int)
MEM_WRITE(1, 0, glue_out_1, OUTPUT_SIZE_LAYER_1 * OUTPUT_SIZE_LAYER_1 * OUTPUT_CHANNELS_LAYER_1)

//Layer 2
#define COARSE_LAYER_2 8

channel int8_t glue_out_2[1] __attribute__((depth(OUTPUT_CHANNELS_LAYER_2)));
channel int8_t csm_out_2[2][1] __attribute__((depth(OUTPUT_CHANNELS_LAYER_2)));
channel int8_t csm_in_2[2] __attribute__((depth(INPUT_CHANNELS_LAYER_2)));

MEM_READ(2, 1, csm_in_2, INPUT_SIZE_LAYER_2 * INPUT_SIZE_LAYER_2 * INPUT_CHANNELS_LAYER_2)
GEN_CSM_LAYER(2, 1, 2, csm_in_2, glue_out_2, COARSE_LAYER_2, int8_t, int)
GEN_FORK_MERGE(1, 2, glue_out_2, csm_out_2, OUTPUT_SIZE_LAYER_2, OUTPUT_SIZE_LAYER_2, OUTPUT_CHANNELS_LAYER_2, int8_t)

//Layer 3
#define COARSE_LAYER_3 4

channel int8_t csm_out_3[2][1] __attribute__((depth(OUTPUT_CHANNELS_LAYER_3)));

GEN_PIPE_CSM_LAYER(1, 1, 3, csm_out_2, csm_out_3, COARSE_LAYER_3, int8_t, int)

//Layer 4
#define COARSE_LAYER_4 4

channel int8_t csm_out_4[2][1] __attribute__((depth(OUTPUT_CHANNELS_LAYER_4)));

GEN_PIPE_CSM_LAYER(1, 1, 4, csm_out_3, csm_out_4, COARSE_LAYER_4, int8_t, int)

//Layer 2 + Layer 4
channel int8_t add_out_0[1] __attribute__((depth(OUTPUT_CHANNELS_LAYER_4)));

GEN_ADD_NONGLOBAL(1, 0, csm_out_4[1], csm_out_4[0], add_out_0, int)

//Layer 5
#define COARSE_LAYER_5 8

channel int8_t csm_out_5[1] __attribute__((depth(OUTPUT_CHANNELS_LAYER_5)));
channel int8_t csm_in_5[2] __attribute__((depth(INPUT_CHANNELS_LAYER_5)));

MEM_READ(2, 2, csm_in_5, INPUT_SIZE_LAYER_5 * INPUT_SIZE_LAYER_5 * INPUT_CHANNELS_LAYER_5)
GEN_CSM_LAYER(2, 1, 5, csm_in_5, csm_out_5, COARSE_LAYER_5, int8_t, int)


//Concat 0
channel int8_t concat_out_0[1] __attribute__((depth(OUTPUT_CHANNELS_LAYER_4 + OUTPUT_CHANNELS_LAYER_5)));

GEN_CONCAT_QUANT_NONGLOBAL(1, 1, 0, csm_out_5, add_out_0, concat_out_0, int8_t)

//Layer 6
#define COARSE_LAYER_6 16

channel int8_t csm_out_6[2] __attribute__((depth(OUTPUT_CHANNELS_LAYER_6 * OUTPUT_SIZE_LAYER_6)));

GEN_CSM_LAYER(1, 2, 6, concat_out_0, csm_out_6, COARSE_LAYER_6, int8_t, int)

//Layer 7
#define COARSE_LAYER_7 8

channel int8_t glue_out_7[1] __attribute__((depth(OUTPUT_CHANNELS_LAYER_7)));
// channel int8_t csm_out_7 __attribute__((depth(OUTPUT_CHANNELS_LAYER_7)));
// channel int8_t csm_in_13 __attribute__((depth(OUTPUT_CHANNELS_LAYER_7)));

GEN_CSM_LAYER(2, 1, 7, csm_out_6, glue_out_7, COARSE_LAYER_7, int8_t, int)
// GEN_FORK_PIXEL(7, glue_out_7, csm_out_7, csm_in_13, OUTPUT_SIZE_LAYER_7, OUTPUT_SIZE_LAYER_7, OUTPUT_CHANNELS_LAYER_7, int8_t)
MEM_WRITE(1, 1, glue_out_7, OUTPUT_SIZE_LAYER_7 * OUTPUT_SIZE_LAYER_7 * OUTPUT_CHANNELS_LAYER_7)

//Layer 8
#define COARSE_LAYER_8 8

channel int8_t glue_out_8[1] __attribute__((depth(OUTPUT_CHANNELS_LAYER_8)));
channel int8_t csm_out_8[2][1] __attribute__((depth(OUTPUT_CHANNELS_LAYER_8)));
channel int8_t csm_in_8[1] __attribute__((depth(INPUT_CHANNELS_LAYER_8)));

MEM_READ(1, 3, csm_in_8, INPUT_SIZE_LAYER_8 * INPUT_SIZE_LAYER_8 * INPUT_CHANNELS_LAYER_8)
GEN_CSM_LAYER(1, 1, 8, csm_in_8, glue_out_8, COARSE_LAYER_8, int8_t, int)
GEN_FORK_MERGE(1, 8, glue_out_8, csm_out_8, OUTPUT_SIZE_LAYER_8, OUTPUT_SIZE_LAYER_8, OUTPUT_CHANNELS_LAYER_8, int8_t)

//Layer 9
#define COARSE_LAYER_9 4

channel int8_t csm_out_9[2][1] __attribute__((depth(OUTPUT_CHANNELS_LAYER_9)));

GEN_PIPE_CSM_LAYER(1, 1, 9, csm_out_8, csm_out_9, COARSE_LAYER_9, int8_t, int)

//Layer 10
#define COARSE_LAYER_10 4

channel int8_t csm_out_10[2][1] __attribute__((depth(OUTPUT_CHANNELS_LAYER_10)));

GEN_PIPE_CSM_LAYER(1, 1, 10, csm_out_9, csm_out_10, COARSE_LAYER_10, int8_t, int)

//Layer 8 + Layer 10
channel int8_t add_out_1_internal[1] __attribute__((depth(OUTPUT_CHANNELS_LAYER_10)));
channel int8_t add_out_1[2][1] __attribute__((depth(OUTPUT_CHANNELS_LAYER_10)));

GEN_ADD_NONGLOBAL(1, 1, csm_out_10[1], csm_out_10[0], add_out_1_internal, int)
GEN_FORK_MERGE(1, add1, add_out_1_internal, add_out_1, HEIGHT_ADD_1, WIDTH_ADD_1, FILTERS_ADD_1, int8_t)

//Layer 11
#define COARSE_LAYER_11 4

channel int8_t csm_out_11[2][1] __attribute__((depth(OUTPUT_CHANNELS_LAYER_11)));

GEN_PIPE_CSM_LAYER(1, 1, 11, add_out_1, csm_out_11, COARSE_LAYER_11, int8_t, int)

//Layer 12
#define COARSE_LAYER_12 4

channel int8_t csm_out_12[2][1] __attribute__((depth(OUTPUT_CHANNELS_LAYER_12)));

GEN_PIPE_CSM_LAYER(1, 1, 12, csm_out_11, csm_out_12, COARSE_LAYER_12, int8_t, int)

//Add 1 + Layer 12
channel int8_t add_out_2[1] __attribute__((depth(OUTPUT_CHANNELS_LAYER_12)));

GEN_ADD_NONGLOBAL(1, 2, csm_out_12[1], csm_out_12[0], add_out_2, int)

//Layer 13
#define COARSE_LAYER_13 8

channel int8_t csm_out_13[1] __attribute__((depth(OUTPUT_CHANNELS_LAYER_13)));
channel int8_t csm_in_13[1] __attribute__((depth(INPUT_CHANNELS_LAYER_13)));

MEM_READ(1, 4, csm_in_13, INPUT_SIZE_LAYER_13 * INPUT_SIZE_LAYER_13 * INPUT_CHANNELS_LAYER_13)
GEN_CSM_LAYER(1, 1, 13, csm_in_13, csm_out_13, COARSE_LAYER_13, int8_t, int)

//Concat 1
channel int8_t concat_out_1[1] __attribute__((depth(OUTPUT_CHANNELS_LAYER_12 + OUTPUT_CHANNELS_LAYER_13)));
GEN_CONCAT_QUANT_NONGLOBAL(1, 1, 1, csm_out_13, add_out_2, concat_out_1, int8_t)

//Layer 14
#define COARSE_LAYER_14 16

channel int8_t glue_out_14[1] __attribute__((depth(OUTPUT_CHANNELS_LAYER_14)));
channel int8_t csm_out_14[1] __attribute__((depth(OUTPUT_CHANNELS_LAYER_14 * OUTPUT_SIZE_LAYER_14)));
channel int8_t global_out_0[1] __attribute__((depth(OUTPUT_CHANNELS_LAYER_14 * OUTPUT_SIZE_LAYER_14)));

GEN_CSM_LAYER(1, 1, 14, concat_out_1, glue_out_14, COARSE_LAYER_14, int8_t, int)
GEN_FORK_PIXEL(1, 14, glue_out_14, csm_out_14, global_out_0, OUTPUT_SIZE_LAYER_14, OUTPUT_SIZE_LAYER_14, OUTPUT_CHANNELS_LAYER_14, int8_t)
MEM_WRITE(1, 2, global_out_0, OUTPUT_SIZE_LAYER_14 * OUTPUT_SIZE_LAYER_14 * OUTPUT_CHANNELS_LAYER_14)

//Layer 15
#define COARSE_LAYER_15 8

channel int8_t glue_out_15[1] __attribute__((depth(OUTPUT_CHANNELS_LAYER_15)));
channel int8_t csm_out_15[1] __attribute__((depth(OUTPUT_CHANNELS_LAYER_15)));
channel int8_t csm_in_23_global[1] __attribute__((depth(OUTPUT_CHANNELS_LAYER_15)));

GEN_CSM_LAYER(1, 1, 15, csm_out_14, glue_out_15, COARSE_LAYER_15, int8_t, int)
GEN_FORK_PIXEL(1, 15, glue_out_15, csm_out_15, csm_in_23_global, OUTPUT_SIZE_LAYER_15, OUTPUT_SIZE_LAYER_15, OUTPUT_CHANNELS_LAYER_15, int8_t)
MEM_WRITE(1, 3, csm_in_23_global, OUTPUT_SIZE_LAYER_15 * OUTPUT_SIZE_LAYER_15 * OUTPUT_CHANNELS_LAYER_15)

//Layer 16
#define COARSE_LAYER_16 8

channel int8_t glue_out_16[1] __attribute__((depth(OUTPUT_CHANNELS_LAYER_16)));
channel int8_t csm_out_16[2][1] __attribute__((depth(OUTPUT_CHANNELS_LAYER_16)));
// channel int8_t csm_in_16 __attribute__((depth(OUTPUT_CHANNELS_LAYER_16)));

// MEM_READ(5, csm_in_16, INPUT_SIZE_LAYER_16 * INPUT_SIZE_LAYER_16 * INPUT_CHANNELS_LAYER_16)
GEN_CSM_LAYER(1, 1, 16, csm_out_15, glue_out_16, COARSE_LAYER_16, int8_t, int)
GEN_FORK_MERGE(1, 16, glue_out_16, csm_out_16, OUTPUT_SIZE_LAYER_16, OUTPUT_SIZE_LAYER_16, OUTPUT_CHANNELS_LAYER_16, int8_t)

//Layer 17
#define COARSE_LAYER_17 4

channel int8_t csm_out_17[2][1] __attribute__((depth(OUTPUT_CHANNELS_LAYER_17)));

GEN_PIPE_CSM_LAYER(1, 1, 17, csm_out_16, csm_out_17, COARSE_LAYER_17, int8_t, int)

//Layer 18
#define COARSE_LAYER_18 4

channel int8_t csm_out_18[2][1] __attribute__((depth(OUTPUT_CHANNELS_LAYER_18)));

GEN_PIPE_CSM_LAYER(1, 1, 18, csm_out_17, csm_out_18, COARSE_LAYER_18, int8_t, int)

//Layer 16 + Layer 18
channel int8_t add_out_3_internal[1] __attribute__((depth(FILTERS_ADD_3)));
channel int8_t add_out_3[2][1] __attribute__((depth(FILTERS_ADD_3)));

GEN_ADD_NONGLOBAL(1, 3, csm_out_18[1], csm_out_18[0], add_out_3_internal, int)
GEN_FORK_MERGE(1, add3, add_out_3_internal, add_out_3, HEIGHT_ADD_3, WIDTH_ADD_3, FILTERS_ADD_3, int8_t)

//Layer 19
#define COARSE_LAYER_19 4

channel int8_t csm_out_19[2][1] __attribute__((depth(OUTPUT_CHANNELS_LAYER_19)));

GEN_PIPE_CSM_LAYER(1, 1, 19, add_out_3, csm_out_19, COARSE_LAYER_19, int8_t, int)

//Layer 20
#define COARSE_LAYER_20 4

channel int8_t csm_out_20[2][1] __attribute__((depth(OUTPUT_CHANNELS_LAYER_20)));

GEN_PIPE_CSM_LAYER(1, 1, 20, csm_out_19, csm_out_20, COARSE_LAYER_20, int8_t, int)

//Add 3 + Layer 20
channel int8_t add_out_4_internal[1] __attribute__((depth(FILTERS_ADD_4)));
// channel int8_t add_out_4[2] __attribute__((depth(FILTERS_ADD_4)));
// channel int8_t add_in_5 __attribute__((depth(HEIGHT_ADD_4 * WIDTH_ADD_4 * FILTERS_ADD_4)));

GEN_ADD_NONGLOBAL(1, 4, csm_out_20[1], csm_out_20[0], add_out_4_internal, int)
// GEN_FORK_MERGE(add4, add_out_4_internal, add_out_4, HEIGHT_ADD_4, WIDTH_ADD_4, FILTERS_ADD_4, int8_t)
MEM_WRITE(1, 4, add_out_4_internal, HEIGHT_ADD_4 * WIDTH_ADD_4 * FILTERS_ADD_4)

// //Layer 21
// #define COARSE_LAYER_21 1

// channel int8_t csm_out_21[2] __attribute__((depth(OUTPUT_CHANNELS_LAYER_21)));
// GEN_PIPE_CSM_LAYER(21, add_out_4, csm_out_21, COARSE_LAYER_21, int8_t, int)

// //Layer 22
// #define COARSE_LAYER_22 1

// channel int8_t csm_out_22[2] __attribute__((depth(OUTPUT_CHANNELS_LAYER_22)));

// GEN_PIPE_CSM_LAYER(22, csm_out_21, csm_out_22, COARSE_LAYER_22, int8_t, int)

// //Add 4 + Layer 22
// channel int8_t add_out_5 __attribute__((depth(FILTERS_ADD_5)));

// GEN_ADD_NONGLOBAL(5, csm_out_22[1], csm_out_22[0], add_out_5, int)

// //Layer 23
// #define COARSE_LAYER_23 2

// channel int8_t csm_in_23 __attribute__((depth(INPUT_CHANNELS_LAYER_23)));
// channel int8_t csm_out_23 __attribute__((depth(OUTPUT_CHANNELS_LAYER_23)));

// MEM_READ(6, csm_in_23, INPUT_SIZE_LAYER_23 * INPUT_SIZE_LAYER_23 * INPUT_CHANNELS_LAYER_23)
// GEN_CSM_LAYER(23, csm_in_23, csm_out_23, COARSE_LAYER_23, int8_t, int)

// //Concat 2
// channel int8_t concat_out_2 __attribute__((depth(FILTERS_ADD_5 + OUTPUT_CHANNELS_LAYER_23)));

// GEN_CONCAT_QUANT_NONGLOBAL(2, csm_out_23, add_out_5, concat_out_2, int8_t)

// //Layer 24
// #define COARSE_LAYER_24 4

// channel int8_t glue_out_24 __attribute__((depth(OUTPUT_CHANNELS_LAYER_24)));
// channel int8_t csm_out_24 __attribute__((depth(OUTPUT_CHANNELS_LAYER_24 * OUTPUT_SIZE_LAYER_24 * 2)));
// channel int8_t global_out_1 __attribute__((depth(OUTPUT_CHANNELS_LAYER_24 * OUTPUT_SIZE_LAYER_24 * 2)));

// GEN_CSM_LAYER(24, concat_out_2, glue_out_24, COARSE_LAYER_24, int8_t, int)
// GEN_FORK_PIXEL(24, glue_out_24, csm_out_24, global_out_1, OUTPUT_SIZE_LAYER_24, OUTPUT_SIZE_LAYER_24, OUTPUT_CHANNELS_LAYER_24, int8_t)
// MEM_WRITE(4, global_out_1, OUTPUT_SIZE_LAYER_24 * OUTPUT_SIZE_LAYER_24 * OUTPUT_CHANNELS_LAYER_24)

// //Layer 25
// #define COARSE_LAYER_25 2

// channel int8_t glue_out_25 __attribute__((depth(OUTPUT_CHANNELS_LAYER_25)));
// // channel int8_t csm_out_25 __attribute__((depth(OUTPUT_CHANNELS_LAYER_25)));
// // channel int8_t csm_in_29 __attribute__((depth(OUTPUT_CHANNELS_LAYER_25)));

// GEN_CSM_LAYER(25, csm_out_24, glue_out_25, COARSE_LAYER_25, int8_t, int)
// MEM_WRITE(5, glue_out_25, OUTPUT_SIZE_LAYER_25 * OUTPUT_SIZE_LAYER_25 * OUTPUT_CHANNELS_LAYER_25)

// //Layer 26
// #define COARSE_LAYER_26 2

// channel int8_t glue_out_26 __attribute__((depth(OUTPUT_CHANNELS_LAYER_26)));
// channel int8_t csm_out_26[2] __attribute__((depth(OUTPUT_CHANNELS_LAYER_26)));

// GEN_CSM_LAYER(26, glue_out_25, glue_out_26, COARSE_LAYER_26, int8_t, int)
// GEN_FORK_MERGE(26, glue_out_26, csm_out_26, OUTPUT_SIZE_LAYER_26, OUTPUT_SIZE_LAYER_26, OUTPUT_CHANNELS_LAYER_26, int8_t)


// //Layer 27
// #define COARSE_LAYER_27 1

// channel int8_t csm_out_27[2] __attribute__((depth(OUTPUT_CHANNELS_LAYER_27)));

// GEN_PIPE_CSM_LAYER(27, csm_out_26, csm_out_27, COARSE_LAYER_27, int8_t, int)

// //Layer 28
// #define COARSE_LAYER_28 1

// channel int8_t csm_out_28[2] __attribute__((depth(OUTPUT_CHANNELS_LAYER_28)));

// GEN_PIPE_CSM_LAYER(28, csm_out_27, csm_out_28, COARSE_LAYER_28, int8_t, int)

// //Layer 26 + Layer 28
// channel int8_t add_out_6 __attribute__((depth(FILTERS_ADD_6)));

// GEN_ADD_NONGLOBAL(6, csm_out_28[1], csm_out_28[0], add_out_6, int)

// //Layer 29
// #define COARSE_LAYER_29 2

// channel int8_t csm_out_29 __attribute__((depth(OUTPUT_CHANNELS_LAYER_29)));
// channel int8_t csm_in_29 __attribute__((depth(INPUT_CHANNELS_LAYER_29)));

// MEM_READ(7, csm_in_29, INPUT_SIZE_LAYER_29 * INPUT_SIZE_LAYER_29 * INPUT_CHANNELS_LAYER_29)
// GEN_CSM_LAYER(29, csm_in_29, csm_out_29, COARSE_LAYER_29, int8_t, int)

// //Concat 3
// channel int8_t concat_out_3 __attribute__((depth(OUTPUT_CHANNELS_LAYER_29 + FILTERS_ADD_6)));

// GEN_CONCAT_QUANT_NONGLOBAL(3, csm_out_29, add_out_6, concat_out_3, int8_t)

// //Layer 30
// #define COARSE_LAYER_30 4

// channel int8_t csm_out_30 __attribute__((depth(OUTPUT_CHANNELS_LAYER_30)));

// GEN_CSM_LAYER(30, concat_out_3, csm_out_30, COARSE_LAYER_30, int8_t, int)

// //Layer 31
// #define COARSE_LAYER_31 2

// channel int8_t csm_out_31 __attribute__((depth(OUTPUT_CHANNELS_LAYER_31)));

// GEN_CSM_LAYER(31, csm_out_30, csm_out_31, COARSE_LAYER_31, int8_t, int)
// MEM_WRITE(5, csm_out_31, OUTPUT_SIZE_LAYER_31 * OUTPUT_SIZE_LAYER_31 * OUTPUT_CHANNELS_LAYER_31)