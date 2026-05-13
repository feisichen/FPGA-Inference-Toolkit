#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <CL/opencl.h>

// user defined library
#include "ocl_util.h"
#include "timer.h"
#include "NetRun.h"
#include "yolov5n_configs.h"

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
#define OUT_BUF_SIZE 20 * 20 * 128
#define DEVICE_NUM 2

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

void load_image(vector<int8_t,  AlignedAllocator<int8_t, 64>>& image, char *image_path)
{
    printf("\nImage preprocessing...\n");

    Mat img = imread(image_path);
    Mat img_resized;

    cv::cvtColor(img, img_resized, cv::COLOR_BGR2RGB);
    letterbox(img_resized, img_resized, cv::Size(640, 640), cv::Scalar(0, 0, 0));

    img_resized.convertTo(img_resized, CV_8U);

    // opencv读取的图片数据已经是HWC格式
    for (int i = 0; i < 3 * 640 * 640; i++)
    {
        uint8_t a = (uint8_t)img_resized.data[i];
        image[i] = (int8_t)a + 128;
    }
}

void load_video(vector<vector<int8_t,  AlignedAllocator<int8_t, 64>>>& video, char *videoPath)
{
    printf("\nVideo preprocessing...\n");
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "无法打开视频文件" << std::endl;
    }

    Mat img;
    Mat img_resized;
    while (true) {
        // 读取视频的下一帧
        bool isSuccess = cap.read(img);

        // 如果读取失败或视频结束，退出循环
        if (!isSuccess) {
            std::cout << "视频结束或读取帧失败" << std::endl;
            break;
        }
        // std::cout << img.size().width << '*' <<img.size().height << std::endl;
        // for(int i=0;i<img.size().width * img.size().height;i++){
        //     std::cout << (int) img.data[i] << std::endl;
        // }
        cv::cvtColor(img, img_resized, cv::COLOR_BGR2RGB);
        letterbox(img_resized, img_resized, cv::Size(640, 640), cv::Scalar(0, 0, 0));

        img_resized.convertTo(img_resized, CV_8U);
        vector<int8_t,  AlignedAllocator<int8_t, 64>> image(3 * 640 * 640);
        // opencv读取的图片数据已经是HWC格式
        for (int i = 0; i < 3 * 640 * 640; i++)
        {
            uint8_t a = (uint8_t)img_resized.data[i];
            image[i] = a + 128;
            // std::cout << (int)image[i] <<std::endl;
        }
        video.push_back(image);
    }   
}

int main(int argc, char **argv)
{
    // std::cout << cv::getBuildInformation() << std::endl;
    cv::setNumThreads(1);
    cl_uint num_devices = 1;
    cl_platform_id platform_id = NULL;
    cl_context context = NULL;
    std::vector<cl_program> programs;

    scoped_array<cl_device_id> device;

    cl_int status;
    unsigned int device_ptr = 0;
    char *image_path = argv[2];

    vector<int8_t,  AlignedAllocator<int8_t, 64>> image0(640*640*3);
    load_image(image0, image_path);
    // vector<int8_t,  AlignedAllocator<int8_t, 64>> image;
    // load_image(video, image_path);
    if (argc != 3)
    {
        printf("Error: wrong commad format, usage:\n");
        printf("%s <binaryfile>\n", argv[0]);
        // return EXIT_FAILURE;
    }

    printf("***************************************************\n");
    printf("PipeCNN: An OpenCL-Based FPGA Accelerator for CNNs \n");
    printf("***************************************************\n");
    // 获取设备
    //  Connect to the desired platform
    platform_id = findPlatform(vendor_name);
    if (platform_id == NULL)
    {
        printf("ERROR: Unable to find the desired OpenCL platform.\n");
        return false;
    }

    // Query the available OpenCL device
    device.reset(getDevices(platform_id, DEVICE_TYPE, &num_devices));
    printf("\nPlatform: %s\n", getPlatformName(platform_id).c_str());
    printf("Totally %d device(s) are found\n", num_devices);

                     // for(unsigned device_ptr = 0; device_ptr < num_devices; ++device_ptr) {
    printf("  Using Device %d: %s\n", device_ptr, getDeviceName(device[device_ptr]).c_str());
    displayDeviceInfo(device[device_ptr]);
    //}
    std::vector<cl_device_id> v_devices;
    for(cl_uint i=0;i<num_devices;i++){
        v_devices.emplace_back(device[i]);
    }

    // 创建上下文对象
    // Create the context.
    context = clCreateContext(NULL, num_devices, &*(v_devices.begin()), NULL, NULL, &status);
    //checkError(status, "Failed to create context");

    // Create Program Objects
    // 创建程序对象，首先获取文件名
    char *kernel_file_name = argv[1];
    printf("\n%s\n", kernel_file_name);

    // Create the program for all device. All devices execute the same kernel.
    programs.push_back(createProgramFromFile(context, "dev0.aocx", &device[0], 1));
    // programs.push_back(createProgramFromFile(context, "dev1.aocx", &device[1], 1));
    std::vector<int> threadnums{20};
    std::vector<Box> result;

    for(auto threadnum:threadnums){
        NetRun<int8_t, void(* )(std::vector<std::reference_wrapper<std::vector<int8_t, AlignedAllocator<int8_t, 64>>>>, std::vector<Box>&)> netrun(yolov5n, yolov5n_pre, yolov5n_weights, CPU_functions, last_function, programs, v_devices, context, threadnum);
        netrun.run();
        result.clear();
        std::this_thread::sleep_for(std::chrono::seconds(4));
        std::vector<int> imagenums{100};
        auto start = chrono::steady_clock::now();
        for(auto imagenum:imagenums){
            for(int i=0;i<imagenum;i++){
                netrun.appendImage(image0);
            }
            for(int i=0;i<imagenum;i++){
                result = netrun.getResult();
                // std:: cout << result[result.size()-1].size() << std::endl;
            }
            auto end = chrono::steady_clock::now();
            auto duration = end - start;
            cout << "time elapsed: " << chrono::duration_cast<chrono::microseconds>(duration).count() / 1000.0 << "ms" << endl;
        }
        netrun.stop();
    }
    for (size_t n = 0; n < result.size(); n++)
	{
		printf("%d | %f | %f | %f | %f | %f\n", result[n].class_id, result[n].x, result[n].y, result[n].width, result[n].height, result[n].conf);
	}
    return 0;
}
