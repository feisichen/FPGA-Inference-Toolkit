#include "type.h"
#include "common.h"
#include "concat.h"
#include "mem_read.h"
#include "mem_write.h"
#include "add.h"
#include "weights.h"
#include "csm.h"
#include "maxpool.h"
#include "fork.h"
#pragma OPENCL EXTENSION cl_intel_channels : enable

//Layer 21
#define COARSE_LAYER_21 2

channel int8_t csm_out_21[2][1] __attribute__((depth(OUTPUT_CHANNELS_LAYER_21)));
channel int8_t csm_in_21[2][1] __attribute__((depth(FILTERS_ADD_4)));
channel int8_t add_out_4_internal[1] __attribute__((depth(FILTERS_ADD_4)));

MEM_READ(1, 5, add_out_4_internal, HEIGHT_ADD_4 * WIDTH_ADD_4 * FILTERS_ADD_4)
GEN_FORK_MERGE(1, add4, add_out_4_internal, csm_in_21, HEIGHT_ADD_4, WIDTH_ADD_4, FILTERS_ADD_4, int8_t)
GEN_PIPE_CSM_LAYER(1, 1, 21, csm_in_21, csm_out_21, COARSE_LAYER_21, int8_t, int)

//Layer 22
#define COARSE_LAYER_22 2

channel int8_t csm_out_22[2][1] __attribute__((depth(OUTPUT_CHANNELS_LAYER_22)));

GEN_PIPE_CSM_LAYER(1, 1, 22, csm_out_21, csm_out_22, COARSE_LAYER_22, int8_t, int)

//Add 4 + Layer 22
channel int8_t add_out_5[1] __attribute__((depth(FILTERS_ADD_5)));

GEN_ADD_NONGLOBAL(1, 5, csm_out_22[1], csm_out_22[0], add_out_5, int)

//Layer 23
#define COARSE_LAYER_23 4

channel int8_t csm_in_23[1] __attribute__((depth(INPUT_CHANNELS_LAYER_23)));
channel int8_t csm_out_23[1] __attribute__((depth(OUTPUT_CHANNELS_LAYER_23)));

MEM_READ(1, 6, csm_in_23, INPUT_SIZE_LAYER_23 * INPUT_SIZE_LAYER_23 * INPUT_CHANNELS_LAYER_23)
GEN_CSM_LAYER(1, 1, 23, csm_in_23, csm_out_23, COARSE_LAYER_23, int8_t, int)

//Concat 2
channel int8_t concat_out_2[1] __attribute__((depth(FILTERS_ADD_5 + OUTPUT_CHANNELS_LAYER_23)));

GEN_CONCAT_QUANT_NONGLOBAL(1, 1, 2, csm_out_23, add_out_5, concat_out_2, int8_t)

//Layer 24
#define COARSE_LAYER_24 8

channel int8_t glue_out_24[1] __attribute__((depth(OUTPUT_CHANNELS_LAYER_24)));
channel int8_t csm_out_24[1] __attribute__((depth(OUTPUT_CHANNELS_LAYER_24 * OUTPUT_SIZE_LAYER_24)));
channel int8_t global_out_1[1] __attribute__((depth(OUTPUT_CHANNELS_LAYER_24 * OUTPUT_SIZE_LAYER_24)));

GEN_CSM_LAYER(1, 1, 24, concat_out_2, glue_out_24, COARSE_LAYER_24, int8_t, int)
GEN_FORK_PIXEL(1, 24, glue_out_24, csm_out_24, global_out_1, OUTPUT_SIZE_LAYER_24, OUTPUT_SIZE_LAYER_24, OUTPUT_CHANNELS_LAYER_24, int8_t)
MEM_WRITE(1, 5, global_out_1, OUTPUT_SIZE_LAYER_24 * OUTPUT_SIZE_LAYER_24 * OUTPUT_CHANNELS_LAYER_24)

//Layer 25
#define COARSE_LAYER_25 4

channel int8_t glue_out_25[1] __attribute__((depth(OUTPUT_CHANNELS_LAYER_25)));
// channel int8_t csm_out_25 __attribute__((depth(OUTPUT_CHANNELS_LAYER_25)));
// channel int8_t csm_in_29 __attribute__((depth(OUTPUT_CHANNELS_LAYER_25)));

GEN_CSM_LAYER(1, 1, 25, csm_out_24, glue_out_25, COARSE_LAYER_25, int8_t, int)
MEM_WRITE(1, 6, glue_out_25, OUTPUT_SIZE_LAYER_25 * OUTPUT_SIZE_LAYER_25 * OUTPUT_CHANNELS_LAYER_25)
//Layer 26
#define COARSE_LAYER_26 4

channel int8_t glue_out_26[1] __attribute__((depth(OUTPUT_CHANNELS_LAYER_26)));
channel int8_t csm_out_26[2][1] __attribute__((depth(OUTPUT_CHANNELS_LAYER_26)));
channel int8_t csm_in_26[1] __attribute__((depth(INPUT_CHANNELS_LAYER_26)));

MEM_READ(1, 7, csm_in_26, INPUT_SIZE_LAYER_26 * INPUT_SIZE_LAYER_26 * INPUT_CHANNELS_LAYER_26)
GEN_CSM_LAYER(1, 1, 26, csm_in_26, glue_out_26, COARSE_LAYER_26, int8_t, int)
GEN_FORK_MERGE(1, 26, glue_out_26, csm_out_26, OUTPUT_SIZE_LAYER_26, OUTPUT_SIZE_LAYER_26, OUTPUT_CHANNELS_LAYER_26, int8_t)


//Layer 27
#define COARSE_LAYER_27 2

channel int8_t csm_out_27[2][1] __attribute__((depth(OUTPUT_CHANNELS_LAYER_27)));

GEN_PIPE_CSM_LAYER(1, 1, 27, csm_out_26, csm_out_27, COARSE_LAYER_27, int8_t, int)

//Layer 28
#define COARSE_LAYER_28 2

channel int8_t csm_out_28[2][1] __attribute__((depth(OUTPUT_CHANNELS_LAYER_28)));

GEN_PIPE_CSM_LAYER(1, 1, 28, csm_out_27, csm_out_28, COARSE_LAYER_28, int8_t, int)

//Layer 26 + Layer 28
channel int8_t add_out_6[1] __attribute__((depth(FILTERS_ADD_6)));

GEN_ADD_NONGLOBAL(1, 6, csm_out_28[1], csm_out_28[0], add_out_6, int)

//Layer 29
#define COARSE_LAYER_29 4

channel int8_t csm_out_29[1] __attribute__((depth(OUTPUT_CHANNELS_LAYER_29)));
channel int8_t csm_in_29[1] __attribute__((depth(INPUT_CHANNELS_LAYER_29)));

MEM_READ(1, 8, csm_in_29, INPUT_SIZE_LAYER_29 * INPUT_SIZE_LAYER_29 * INPUT_CHANNELS_LAYER_29)
GEN_CSM_LAYER(1, 1, 29, csm_in_29, csm_out_29, COARSE_LAYER_29, int8_t, int)

//Concat 3
channel int8_t concat_out_3[1] __attribute__((depth(OUTPUT_CHANNELS_LAYER_29 + FILTERS_ADD_6)));

GEN_CONCAT_QUANT_NONGLOBAL(1, 1, 3, csm_out_29, add_out_6, concat_out_3, int8_t)

//Layer 30
#define COARSE_LAYER_30 8

channel int8_t csm_out_30[1] __attribute__((depth(OUTPUT_CHANNELS_LAYER_30)));

GEN_CSM_LAYER(1, 1, 30, concat_out_3, csm_out_30, COARSE_LAYER_30, int8_t, int)

//Layer 31
#define COARSE_LAYER_31 4

channel int8_t csm_out_31[1] __attribute__((depth(OUTPUT_CHANNELS_LAYER_31)));

GEN_CSM_LAYER(1, 1, 31, csm_out_30, csm_out_31, COARSE_LAYER_31, int8_t, int)
MEM_WRITE(1, 7, csm_out_31, OUTPUT_SIZE_LAYER_31 * OUTPUT_SIZE_LAYER_31 * OUTPUT_CHANNELS_LAYER_31)
