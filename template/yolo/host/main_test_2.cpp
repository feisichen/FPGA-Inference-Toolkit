#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <iostream>
#include <fstream>

#include <chrono>

#include <CL/opencl.h>

// user defined library
#include "ocl_util.h"
#include "timer.h"

// CNN network configuration file
//#include "../device/hw_param.cl"
//#include "layer_config.h"

#ifdef USE_OPENCV
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
using namespace cv;
#endif

using namespace std;
using namespace ocl_util;

#define DEVICE_TYPE CL_DEVICE_TYPE_ACCELERATOR
#define MAX_BATCH_SIZE 16
#define IN_BUF_SIZE 640 * 640 * 3
#define OUT_BUF_SIZE 20 * 20 * 256

//----------- Design Parameters --------------//
// select what platform is used
#ifdef XILINX
const char *vendor_name = "Xilinx";
#else
//----------- SDK version <= 19.1 -----------//
// const char *vendor_name = "Intel";
//----------- SDK version >= 19.3 ----------//
#if defined(SW_EMU)
// const char *vendor_name = "Intel(R) FPGA Emulation Platform for OpenCL(TM)";
 const char *vendor_name = "Intel(R) FPGA SDK for OpenCL(TM)";
#else
const char *vendor_name = "Intel(R) FPGA SDK for OpenCL(TM)";
#endif
#endif

#define KNL_MEM_WR_NAME(ch_num) \
    const char *knl_name_mem_wr_##ch_num = "mem_write_" #ch_num;

#define KNL_MEM_WR_PTR(ch_num)                   \
    scoped_array<cl_kernel> knl_mem_wr_##ch_num; \
    scoped_array<cl_command_queue> que_mem_wr_##ch_num;

#define KNL_MEM_WR_VARI_INIT(ch_num) \
    KNL_MEM_WR_NAME(ch_num)          \
    KNL_MEM_WR_PTR(ch_num)

#define MEM_WR_PTR_RST(ch_num, num_devices) \
    knl_mem_wr_##ch_num.reset(num_devices); \
    que_mem_wr_##ch_num.reset(num_devices);

#define MEM_WR_KERNEL_INIT(program, ch_num, device_num, status)                                                     \
    que_mem_wr_##ch_num[device_num] = clCreateCommandQueue(context, device[0], CL_QUEUE_PROFILING_ENABLE, &status); \
    checkError(status, "Failed to create command queue 1");                                                         \
    knl_mem_wr_##ch_num[device_num] = clCreateKernel(program, knl_name_mem_wr_##ch_num, &status);                   \
    checkError(status, "Failed to create mem_wr_" #ch_num " kernel");

#define MEM_WR_EVT_INIT(ch_num, num_devices) \
    scoped_array<cl_event> mem_wr_event_##ch_num(num_devices);


const char *knl_name_mem_rd = "mem_read_0";
// const char *knl_name_mem_wr = "mem_write";

cl_uint num_devices = 0;
cl_platform_id platform_id = NULL;
cl_context context = NULL;
cl_program program_0 = NULL;
cl_program program_1 = NULL;

scoped_array<cl_device_id> device;
scoped_array<cl_kernel> knl_mem_rd;
scoped_array<cl_kernel> knl_mem_wr;

scoped_array<cl_command_queue> que_mem_rd;
scoped_array<cl_command_queue> que_mem_wr;

scoped_array<cl_mem> data_buf;
scoped_array<cl_mem> output_buf;

enum input_item
{

    image_w,
    image_h,
    image_n, // original image size

    batch_size

};

typedef struct Layer_info
{
    int input_channels;
    int input_size;
    int kernel_size;
    int pad;
    int stride;
    int output_size;
    int output_channels;
    int *bias_conv;
    int8_t *zero_point_w;
    int8_t *zero_point_b;
    int8_t zero_point_i;
    int8_t zero_point_o;
    int *Num_conv;
    int *N_conv;
    int8_t zero_point_i_sig;
    int8_t zero_point_o_mul;
    int Num_mul;
    int N_mul;
} Layer_info;

typedef struct Layer0_info
{
    int input_channels = 3;
    int input_size = 640;
    int kernel_size = 6;
    int pad = 0;
    int stride = 2;
    int output_size = 320;
    int output_channels = 16;
    int bias_conv[16] = {35658, 40078, 6242, 8123, 18472, 30770, 25205, 32351, 10660, 37843, 75210, 54084, 13691, 10497, 11288, 9193};
    int8_t zero_point_w[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int8_t zero_point_b[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int8_t zero_point_i = -128;
    int8_t zero_point_o = 0;
    int Num_conv[16] = {555, 307, 3713, 4089, 1573, 773, 447, 1441, 4555, 375, 979, 1125, 1667, 245, 1703, 2507};
    int N_conv[16] = {21, 21, 23, 22, 23, 22, 21, 23, 22, 21, 23, 23, 21, 18, 21, 21};
    // Mul
    int8_t zero_point_i_sig = -128;
    int8_t zero_point_o_mul = -126;
    int Num_mul = 8191;
    int N_mul = 20;
} Layer0_info;

void cleanup()
{
    cout << "cleanup" << endl;
}

// 以填充方式调整图像大小-->640*640
void letterbox(const cv::Mat &image, cv::Mat &outImage,
               const cv::Size &newShape = cv::Size(640, 640),
               const cv::Scalar &color = cv::Scalar(114, 114, 114))
{
    cv::Size shape = image.size();
    float r = std::min((float)newShape.height / (float)shape.height,
                       (float)newShape.width / (float)shape.width);

    int newUnpad[2]{(int)std::round((float)shape.width * r),
                    (int)std::round((float)shape.height * r)};

    auto dw = (float)(newShape.width - newUnpad[0]);
    auto dh = (float)(newShape.height - newUnpad[1]);

    dw /= 2.0f;
    dh /= 2.0f;

    if (shape.width != newUnpad[0] && shape.height != newUnpad[1])
    {
        cv::resize(image, outImage, cv::Size(newUnpad[0], newUnpad[1]));
    }

    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));
    cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}

void load_image(int8_t *image, struct Layer0_info *layer, char *image_path)
{
    printf("\nImage preprocessing...\n");

    Mat img = imread(image_path);
    Mat img_resized;

    cv::cvtColor(img, img_resized, cv::COLOR_BGR2RGB);
    letterbox(img_resized, img_resized, cv::Size(640, 640), cv::Scalar(0, 0, 0));
    // resize(img, img_resized, Size(640,640), INTER_LINEAR);

    img_resized.convertTo(img_resized, CV_8U);

    // opencv读取的图片数据已经是HWC格式
    for (int i = 0; i < 3 * 640 * 640; i++)
    {
        uint8_t a = (uint8_t)img_resized.data[i];
        image[i] = a - layer->zero_point_i;
    }
}

template <typename T, typename U>
static void assign(T *l, U *r)
{
    l->input_channels = r->input_channels;
    l->input_size = r->input_size;
    l->kernel_size = r->kernel_size;
    l->pad = r->pad;
    l->stride = r->stride;
    l->output_size = r->output_size;
    l->output_channels = r->output_channels;
    l->bias_conv = r->bias_conv;
    l->zero_point_w = r->zero_point_w;
    l->zero_point_b = r->zero_point_b;
    l->zero_point_i = r->zero_point_i;
    l->zero_point_o = r->zero_point_o;
    l->Num_conv = r->Num_conv;
    l->N_conv = r->N_conv;
    l->zero_point_i_sig = r->zero_point_i_sig;
    l->zero_point_o_mul = r->zero_point_o_mul;
    l->Num_mul = r->Num_mul;
    l->N_mul = r->N_mul;
}

int main(int argc, char **argv)
{
    cl_int status;
    unsigned int device_ptr;
    char *image_path = argv[2];
    struct Layer0_info layer0;
    struct Layer_info layer;
    assign(&layer, &layer0);
    auto t0 = chrono::steady_clock::now();
    int8_t *image0 = (int8_t *)aligned_alloc(64 ,layer.input_size * layer.input_size * layer.input_channels * sizeof(int8_t));
    load_image(image0, &layer0, image_path);
    if (argc != 3) {
        printf("Error: wrong commad format, usage:\n");
        printf("%s <binaryfile>\n", argv[0]);
        return EXIT_FAILURE;
    }

    printf("***************************************************\n");
    printf("PipeCNN: An OpenCL-Based FPGA Accelerator for CNNs \n");
    printf("***************************************************\n");
    // 获取设备
    //  Connect to the desired platform
    platform_id = findPlatform(vendor_name);
    if (platform_id == NULL) {
        printf("ERROR: Unable to find the desired OpenCL platform.\n");
        return false;
    }

    // Query the available OpenCL device
    device.reset(getDevices(platform_id, DEVICE_TYPE, &num_devices));
    printf("\nPlatform: %s\n", getPlatformName(platform_id).c_str());
    printf("Totally %d device(s) are found\n", num_devices);

    if (num_devices < 2) {
        printf("ERROR: Require two devices.\n");
        return EXIT_FAILURE;
    }

    for(unsigned device_ptr = 0; device_ptr < num_devices; ++device_ptr) {
        printf("  Using Device %d: %s\n", device_ptr, getDeviceName(device[device_ptr]).c_str());
        displayDeviceInfo(device[device_ptr]);
    }

    // 创建上下文对象
    // Create the context.
    context = clCreateContext(NULL, num_devices, device, NULL, NULL, &status);
    checkError(status, "Failed to create context");

    // Create the program for device.
    program_0 = createProgramFromFile(context, "dev0.aocx", &device[0], 1);
    program_1 = createProgramFromFile(context, "dev1.aocx", &device[1], 1);

    knl_mem_wr.reset(5);
    que_mem_wr.reset(5);

    que_mem_rd.reset(3);
    knl_mem_rd.reset(3);

    data_buf.reset(1);
    output_buf.reset(5);

    printf("init\n");

    cout << "cmd que, kernel init\n";
    que_mem_rd[0] = clCreateCommandQueue(context, device[0], CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status, "Failed to create command queue for mem_read_0");
    knl_mem_rd[0] = clCreateKernel(program_0, "mem_read_0", &status);
    checkError(status, "Failed to create kernel mem_read_0");

    que_mem_rd[1] = clCreateCommandQueue(context, device[1], CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status, "Failed to create command queue for mem_read_1");
    knl_mem_rd[1] = clCreateKernel(program_1, "mem_read_1", &status);
    checkError(status, "Failed to create kernel mem_read_1");

    que_mem_rd[2] = clCreateCommandQueue(context, device[1], CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status, "Failed to create command queue for mem_read_2");
    knl_mem_rd[2] = clCreateKernel(program_1, "mem_read_2", &status);
    checkError(status, "Failed to create kernel mem_read_2");

    que_mem_wr[0] = clCreateCommandQueue(context, device[0], CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status, "Failed to create command queue for mem_write_0");
    knl_mem_wr[0] = clCreateKernel(program_0, "mem_write_0", &status);
    checkError(status, "Failed to create kernel mem_write_0");

    que_mem_wr[1] = clCreateCommandQueue(context, device[0], CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status, "Failed to create command queue for mem_write_1");
    knl_mem_wr[1] = clCreateKernel(program_0, "mem_write_1", &status);
    checkError(status, "Failed to create kernel mem_write_1");

    que_mem_wr[2] = clCreateCommandQueue(context, device[0], CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status, "Failed to create command queue for mem_write_2");
    knl_mem_wr[2] = clCreateKernel(program_0, "mem_write_2", &status);
    checkError(status, "Failed to create kernel mem_write_2");

    que_mem_wr[3] = clCreateCommandQueue(context, device[1], CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status, "Failed to create command queue for mem_write_3");
    knl_mem_wr[3] = clCreateKernel(program_1, "mem_write_3", &status);
    checkError(status, "Failed to create kernel mem_write_3");

    que_mem_wr[4] = clCreateCommandQueue(context, device[1], CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status, "Failed to create command queue for mem_write_4");
    knl_mem_wr[4] = clCreateKernel(program_1, "mem_write_4", &status);
    checkError(status, "Failed to create kernel mem_write_4");

    cout << "buf init\n";
    data_buf[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, IN_BUF_SIZE * sizeof(cl_char), image0, &status);
    checkError(status, "Failed to create buffer for input");

    output_buf[0] = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * 80 * 80 * sizeof(cl_char), nullptr, &status);
    checkError(status, "Failed to create buffer for output0");
    output_buf[1] = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * 40 * 40 * sizeof(cl_char), nullptr, &status);
    checkError(status, "Failed to create buffer for output1");
    output_buf[2] = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * 40 * 40 * sizeof(cl_char), nullptr, &status);
    checkError(status, "Failed to create buffer for output2");
    output_buf[3] = clCreateBuffer(context, CL_MEM_READ_WRITE, 128 * 40 * 40 * sizeof(cl_char), nullptr, &status);
    checkError(status, "Failed to create buffer for output3");
    output_buf[4] = clCreateBuffer(context, CL_MEM_WRITE_ONLY, OUT_BUF_SIZE * sizeof(cl_char), nullptr, &status);
    checkError(status, "Failed to create buffer for output4");


    // cout << "wr buf\n";
    // status = clEnqueueWriteBuffer(que_mem_rd[0], data_buf[0], CL_TRUE, 0, layer.input_size * layer.input_size * layer.input_channels * sizeof(cl_char), image0, 0, nullptr, nullptr);
    // checkError(status, "Failed to copy data");

    // clFinish(que_mem_rd[0]);

    // printf("fin init\n");

    scoped_array<cl_event> mem_rd_event(3);

    scoped_array<cl_event> mem_wr_event(5);

    unsigned argi = 0;
    cl_char *str_rd_1 = new cl_char[OUT_BUF_SIZE];

    printf("set args\n");
    status = clSetKernelArg(knl_mem_rd[0], argi, sizeof(cl_char *), &data_buf[0]);
    checkError(status, "Failed to set argument %d of kernel mem_read_0", argi);

    status = clSetKernelArg(knl_mem_wr[0], argi, sizeof(cl_char *), &output_buf[0]);
    checkError(status, "Failed to set argument %d of kernel mem_write_0", argi);

    status = clSetKernelArg(knl_mem_wr[1], argi, sizeof(cl_char *), &output_buf[1]);
    checkError(status, "Failed to set argument %d of kernel mem_write_1", argi);

    status = clSetKernelArg(knl_mem_wr[2], argi, sizeof(cl_char *), &output_buf[2]);
    checkError(status, "Failed to set argument %d of kernel mem_write_2", argi);

    status = clSetKernelArg(knl_mem_rd[1], argi, sizeof(cl_char *), &output_buf[2]);
    checkError(status, "Failed to set argument %d of kernel mem_read_0", argi);

    status = clSetKernelArg(knl_mem_rd[2], argi, sizeof(cl_char *), &output_buf[1]);
    checkError(status, "Failed to set argument %d of kernel mem_read_0", argi);

    status = clSetKernelArg(knl_mem_wr[3], argi, sizeof(cl_char *), &output_buf[3]);
    checkError(status, "Failed to set argument %d of kernel mem_write_3", argi);

    status = clSetKernelArg(knl_mem_wr[4], argi, sizeof(cl_char *), &output_buf[4]);
    checkError(status, "Failed to set argument %d of kernel mem_write_4", argi);


    printf("enqueue task\n");
    status = clEnqueueTask(que_mem_rd[0], knl_mem_rd[0], 0, nullptr, &mem_rd_event[0]);
    checkError(status, "Failed to launch kernel mem_read_0");

    status = clEnqueueTask(que_mem_wr[0], knl_mem_wr[0], 0, nullptr, &mem_wr_event[0]);
    checkError(status, "Failed to launch kernel mem_write_0");
    status = clEnqueueTask(que_mem_wr[1], knl_mem_wr[1], 0, nullptr, &mem_wr_event[1]);
    checkError(status, "Failed to launch kernel mem_write_1");
    status = clEnqueueTask(que_mem_wr[2], knl_mem_wr[2], 0, nullptr, &mem_wr_event[2]);
    checkError(status, "Failed to launch kernel mem_write_2");

    status = clEnqueueTask(que_mem_rd[1], knl_mem_rd[1], 1, &mem_wr_event[2], &mem_rd_event[1]);
    checkError(status, "Failed to launch kernel mem_read_1");
    status = clEnqueueTask(que_mem_rd[2], knl_mem_rd[2], 1, &mem_wr_event[1], &mem_rd_event[2]);
    checkError(status, "Failed to launch kernel mem_read_2");

    status = clEnqueueTask(que_mem_wr[3], knl_mem_wr[3], 0, nullptr, &mem_wr_event[3]);
    checkError(status, "Failed to launch kernel mem_write_3");
    status = clEnqueueTask(que_mem_wr[4], knl_mem_wr[4], 0, nullptr, &mem_wr_event[4]);
    checkError(status, "Failed to launch kernel mem_write_4");

    auto t1 = chrono::steady_clock::now();

    printf("wait for events\n");
    // status = clWaitForEvents(num_devices, mem_rd_event);
    // checkError(status, "Failed to finish mem_read events");

    // printf("wait for wr0 events\n");
    // status = clWaitForEvents(num_devices, mem_wr_event_0);
    // checkError(status, "Failed to finish wr0");
    // printf("wait for wr1 events\n");
    // status = clWaitForEvents(num_devices, mem_wr_event_1);
    // checkError(status, "Failed to finish wr1");
    // printf("wait for wr2 events\n");
    // status = clWaitForEvents(num_devices, mem_wr_event_2);
    // checkError(status, "Failed to finish wr2");
    // printf("wait for wr3 events\n");
    // status = clWaitForEvents(num_devices, mem_wr_event_3);
    // checkError(status, "Failed to finish wr3");
    printf("wait for mem_write_4 event\n");
    status = clWaitForEvents(1, &mem_wr_event[4]);
    checkError(status, "Failed to finish mem_write_4");

    auto t2 = chrono::steady_clock::now();

    printf("host read&print\n");
    status = clEnqueueReadBuffer(que_mem_wr[4], output_buf[4], CL_TRUE, // read from device0
                                    0, sizeof(cl_char) * OUT_BUF_SIZE, (void *)str_rd_1, 0, NULL, nullptr);
    checkError(status, "Failed to set transfer output data");
    auto t3 = chrono::steady_clock::now();
    for (int i = 0; i < OUT_BUF_SIZE; ++i) {
        cout << (long long)str_rd_1[i] << endl;
    }


    printf("release event\n");
    clReleaseMemObject(data_buf[0]);
    
    for(int i = 0; i < 5; ++i) {
        clReleaseEvent(mem_wr_event[i]);
        clReleaseMemObject(output_buf[i]);
        clReleaseCommandQueue(que_mem_wr[i]);
        clReleaseKernel(knl_mem_wr[i]);
    }
 
    for(int i = 0; i < 3; ++i) {
        clReleaseEvent(mem_rd_event[i]);
        clReleaseCommandQueue(que_mem_rd[i]);
        clReleaseKernel(knl_mem_rd[i]);
    }

    clReleaseProgram(program_0);
    clReleaseProgram(program_1);

    clReleaseContext(context);
    
    auto duration = t1 - t0;
    cout << "prepare time: " << chrono::duration_cast<chrono::microseconds>(duration).count() / 1000.0 << "ms" << endl;
    duration = t2 - t1;
    cout << "fpga execution time: " << chrono::duration_cast<chrono::microseconds>(duration).count() / 1000.0 << "ms" << endl;
    duration = t3 - t2;
    cout << "read buffer time: " << chrono::duration_cast<chrono::microseconds>(duration).count() / 1000.0 << "ms" << endl;
    duration = t3 - t0;
    cout << "total: " << chrono::duration_cast<chrono::microseconds>(duration).count() / 1000.0 << "ms" << endl;


    delete[] str_rd_1;
    free(image0);
    // delete[] str_rd_2;
    // delete[] str_rd_3;

    return 0;
}