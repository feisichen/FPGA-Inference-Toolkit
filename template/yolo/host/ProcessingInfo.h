#ifndef PROCESSING_INFO_H
#define PROCESSING_INFO_H

#include "Layer.h"
#include <CL/opencl.h>
#include "ocl_util.h"
#include "timer.h"
#include <vector>
#include <unordered_map>
#include <utility>
#include <memory>
#include "NetRun.h"
#include "Mem.h"
#include <mutex>
#include <future>
#include <atomic>
#include <list>
#include "Logger.h"
#include "AlignedAllocator.h"
#include "output_type_trait.h"

using namespace ocl_util;

template<typename T, typename Last_function>
class NetRun;

template<typename T, typename Last_function>
class ProcessingInfo;

template<typename T>
class Mem;

template<typename T, typename Last_function>
class ProcessingInfo: public std::enable_shared_from_this<ProcessingInfo<T, Last_function>> {
    friend class NetRun<T, Last_function>;
public:
    using output_type = typename output_type_trait<Last_function>::output_type;
    ProcessingInfo(NetRun<T, Last_function>* netRun, const std::vector<T,  AlignedAllocator<T, 64>>& input_image):netRun_(netRun){
        for(auto& input : netRun_->net_[0].inputs){
            if(input.input_name == "input"){
                Mem_map_["input"] = std::make_shared<Mem<std::vector<T, AlignedAllocator<T, 64>>>>(HOST, input_image, input.input_size);
                break;
            }
        }
        if(netRun_->net_[0].device_type == DEVICE){
            cl_int status;
            for(const auto &output:netRun_->net_[0].outputs){
                cl_mem temp_out = clCreateBuffer(netRun_->context_, CL_MEM_READ_WRITE, output.output_size, nullptr, &status);
                checkError(status, "Failed to create buffer for Layer%d output", 0);
                Mem_map_[output.output_name] = std::make_shared<Mem<std::vector<T, AlignedAllocator<T, 64>>>>(DEVICE, temp_out, output.output_size);
            }
        }
        else{
            for(const auto &output:netRun_->net_[0].outputs){
                Mem_map_[output.output_name] = std::make_shared<Mem<std::vector<T, AlignedAllocator<T, 64>>>>(HOST, std::move(std::vector<T, AlignedAllocator<T, 64>>(output.output_size)), output.output_size);
            }
        }
    }
    ProcessingInfo(NetRun<T, Last_function>* netRun, std::vector<T,  AlignedAllocator<T, 64>>&& input_image):netRun_(netRun){
        for(auto& input : netRun_->net_[0].inputs){
            if(input.input_name == "input"){
                Mem_map_["input"] = std::make_shared<Mem<std::vector<T, AlignedAllocator<T, 64>>>>(HOST, std::move(input_image), input.input_size);
                break;
            }
        }
        if(netRun_->net_[0].device_type == DEVICE){
            cl_int status;
            for(const auto &output:netRun_->net_[0].outputs){
                cl_mem temp_out = clCreateBuffer(netRun_->context_, CL_MEM_READ_WRITE, output.output_size, nullptr, &status);
                checkError(status, "Failed to create buffer for Layer%d output", 0);
                Mem_map_[output.output_name] = std::make_shared<Mem<std::vector<T, AlignedAllocator<T, 64>>>>(DEVICE, temp_out, output.output_size);
            }
        }
        else{
            for(const auto &output:netRun_->net_[0].outputs){
                Mem_map_[output.output_name] = std::make_shared<Mem<std::vector<T, AlignedAllocator<T, 64>>>>(HOST, std::move(std::vector<T, AlignedAllocator<T, 64>>(output.output_size)), output.output_size);
            }
        }
    }
    void commitRequest(){
        if(net_idx_ < netRun_->net_.size()){
            if(netRun_->net_[net_idx_].device_type == DEVICE){
                LOG_DEBUG("Add new device task:%s!", netRun_->net_[net_idx_].name.c_str());
                netRun_->device_request_queue_[netRun_->net_[net_idx_].name].push_back(std::enable_shared_from_this<ProcessingInfo<T, Last_function>>::shared_from_this());
            }
            else{
                LOG_DEBUG("Add new host task!");
                netRun_->host_request_queue_.push_back(std::enable_shared_from_this<ProcessingInfo<T, Last_function>>::shared_from_this());
            }
        }
        else{
            LOG_DEBUG("Add new final task!");
            netRun_->final_request_queue_.push(std::enable_shared_from_this<ProcessingInfo<T, Last_function>>::shared_from_this());
        }
    }
    void doNext(){
        // if(netRun_->net_[net_idx_].name == "maxpool_2"){
        //     for(auto elem:(Mem_map_[netRun_->net_[net_idx_].outputs[0].output_name])->host_mem){
        //         std::cout << int(elem) << std::endl;
        //     }
        // }
        net_idx_++;
        LOG_DEBUG("Now at Layer:%lu", net_idx_);
        commitRequest();
    }
    std::vector<output_type> getFinalResult(){
        LOG_DEBUG("getFinalResult start!\n");
        for(auto& event:m_outputs_events_){
            clWaitForEvents(1, &event);
        }
        std::vector<output_type> result;
        if(m_outputs_->mem_type == DEVICE){
            std::vector<output_type> temp(m_outputs_->m_size);
            cl_int status;
            status = clEnqueueReadBuffer(netRun_->queues_["buffer_rw"], m_outputs_->device_mem, CL_TRUE, 0, sizeof(T) * m_outputs_->m_size, &*(temp.begin()), 0, NULL, nullptr);
            // clFinish(queues_["buffer_rw"]);
            checkError(status, "Failed to read buffer from result");
           return temp;
        }
        else{
            return m_outputs_->host_mem;
        }
    }
private:
    NetRun<T, Last_function>* netRun_;
    size_t net_idx_ = 0;
    size_t outputs_finished_ = 0;
    //listen
    std::unordered_map<std::string, std::shared_ptr<Mem<std::vector<T, AlignedAllocator<T, 64>>>>> Mem_map_;
    std::unordered_map<std::string, cl_event> Event_map_;
    std::shared_ptr<Mem<std::vector<output_type>>> m_outputs_;
    std::vector<cl_event> m_outputs_events_;
};

#endif