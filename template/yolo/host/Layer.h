#ifndef Layer_H
#define Layer_H
#include <string>
#include <vector>

enum DeviceType{ NONE, DEVICE, HOST, ALL };

struct Input{
    std::string input_name;
    unsigned input_size;
};

struct Output{
    std::string output_name;
    unsigned output_size;
};

struct Layer{
    std::string name;
    DeviceType device_type;
    union{
        int device_index;
        int threads_num;
    };
    std::vector<Input> inputs;
    std::vector<Output> outputs;
};

struct Pre_Layer{
    std::string name;
    DeviceType device_type;
    union{
        int device_index;
        int threads_num;
    };
    std::vector<Input> inputs;
};
#endif