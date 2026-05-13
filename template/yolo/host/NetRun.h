#ifndef NET_RUN_H
#define NET_RUN_H

#include "Queue.h"
#include <list>
#include <mutex>
#include <atomic>
#include <CL/opencl.h>
#include "ocl_util.h"
#include "timer.h"
#include <string>
#include <unordered_map>
#include <unordered_set>
#include "Layer.h"
#include <thread>
#include <vector>
#include <memory>
#include "Mem.h"
#include "ProcessingInfo.h"
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
class NetRun{
    friend class ProcessingInfo<T, Last_function>;
private:
    using Net = std::vector<Layer>;
    using PreNet = std::vector<Pre_Layer>;
    using Host_function = void(*)(int, int, std::vector<std::reference_wrapper<std::vector<T, AlignedAllocator<T, 64>>>> , std::vector<std::reference_wrapper<std::vector<T, AlignedAllocator<T, 64>>>>);
    NetRun(const NetRun&) = delete;
    NetRun& operator=(const NetRun&) = delete;

    //listen main竞争
    threadsafe_queue<std::vector<T,  AlignedAllocator<T, 64>>> image_queue_;

    //listen
    std::list<std::shared_ptr<ProcessingInfo<T, Last_function>>> processinging_queue_;

    std::unordered_map<std::string, cl_kernel> kernels_;
    std::unordered_map<std::string, cl_command_queue> queues_;
    std::unordered_map<std::string, cl_event> events_;
 
    std::unordered_map<std::string, std::vector<std::string>> kernels_inputs_;
    std::unordered_map<std::string, std::vector<std::string>> kernels_outputs_;
    //listen
    std::unordered_map<std::string, std::list<std::shared_ptr<ProcessingInfo<T, Last_function>>>> device_request_queue_;
    std::unordered_map<std::string, size_t> device_request_idx_;
    std::list<std::shared_ptr<ProcessingInfo<T, Last_function>>> host_request_queue_;
    //listen calculate竞争
    threadsafe_queue<std::packaged_task<void()>> host_task_queue_;


    //listen
    // size_t final_layers_ = 0;
    
    threadsafe_queue<std::shared_ptr<ProcessingInfo<T, Last_function>>> final_request_queue_;

    threadsafe_queue<std::shared_ptr<ProcessingInfo<T, Last_function>>> donext_queue_;

    std::vector<std::thread> thread_pool_;

    bool stop_ = true;

    Net net_;
    PreNet pre_net_;
    std::unordered_map<std::string, std::shared_ptr<Mem<std::vector<T, AlignedAllocator<T, 64>>>>> net_weights_;
    std::unordered_map<std::string, Host_function> functions_;
    std::unordered_map<std::string, uint> mem_life_end_;
    Last_function last_function_;
    const std::vector<cl_program>& program_;
    const std::vector<cl_device_id>& device_;
    const cl_context& context_;
    unsigned thread_num_;
    void popProcessing();
    void calculate();
    void listen();
    bool start_host_function(std::shared_ptr<ProcessingInfo<T, Last_function>> processing);
    void start_kernel(std::shared_ptr<ProcessingInfo<T, Last_function>> processing, std::list<std::shared_ptr<ProcessingInfo<T, Last_function>>>& donext_list);
    
    template<typename Arg>
    void add_arg(unsigned &arg_index, const std::string& kernel_name, Arg& kernel_arg);
public:
    using last_function_output_type = typename output_type_trait<Last_function>::output_type;
    NetRun(const Net& net, const PreNet& pre_net, const std::unordered_map<std::string, std::vector<T, AlignedAllocator<T, 64>>>& net_weights_, const std::unordered_map<std::string, Host_function>& functions, Last_function last_function, const std::vector<cl_program>& program, const std::vector<cl_device_id>& device, const cl_context& context, int thread_num = 0);
    ~NetRun();
    void appendImage(const std::vector<T,  AlignedAllocator<T, 64>>& image);
    std::vector<last_function_output_type> getResult();
    bool getResult(std::vector<std::vector<T,  AlignedAllocator<T, 64>>>&);
    void run();
    void stop();
};

template<typename T, typename Last_function>
bool NetRun<T, Last_function>::start_host_function(std::shared_ptr<ProcessingInfo<T, Last_function>> processing){
    std::string func_name = net_[processing->net_idx_].name;
    // LOG_DEBUG("start_host_function:%s!", func_name.c_str());
    int thread_num = net_[processing->net_idx_].threads_num;
    std::unordered_map<std::string, std::shared_ptr<Mem<std::vector<T, AlignedAllocator<T, 64>>>>> lock_rmems;
    std::unordered_map<std::string, std::shared_ptr<Mem<std::vector<T, AlignedAllocator<T, 64>>>>> lock_wmems;
    if(processing->net_idx_ < net_.size() - 1){
        Host_function func = functions_[func_name];
        std::shared_ptr<std::vector<std::future<void>>> real_futures = std::make_shared<std::vector<std::future<void>>>();
        std::vector<std::future<void>>& futures = *real_futures;
        std::vector<std::reference_wrapper<std::vector<T, AlignedAllocator<T, 64>>>> inputs;
        for(auto& input:net_[processing->net_idx_].inputs){
            if(input.input_name.substr(0, 7) == "weights"){
                LOG_DEBUG("add weights:%s!", input.input_name.c_str());
                auto it = net_weights_.find(input.input_name);
                if(it != processing->Mem_map_.end()) {
                    std::shared_ptr<Mem<std::vector<T, AlignedAllocator<T, 64>>>> temp_mem = net_weights_[input.input_name];
                    inputs.emplace_back(temp_mem->host_mem);
                }
                else{
                    for(auto& e_mem:lock_rmems){
                        e_mem.second->rw_mutex.unlock_shared();
                    }
                    LOG_FATAL("no such weights, which name:%s", input.input_name.c_str());
                    return false;
                }
            }
            else{
                auto it = processing->Mem_map_.find(input.input_name);
                if(it != processing->Mem_map_.end()) {
                    std::shared_ptr<Mem<std::vector<T, AlignedAllocator<T, 64>>>> temp_mem = processing->Mem_map_[input.input_name];
                    if(temp_mem->rw_mutex.try_lock_shared()){
                        lock_rmems[input.input_name] = temp_mem;
                        if(temp_mem->mem_type == DEVICE){
                            cl_int status;
                            temp_mem->host_mem.resize(input.input_size);
                            status = clEnqueueReadBuffer(queues_["buffer_rw"], temp_mem->device_mem, CL_TRUE, 0, sizeof(T) * input.input_size, &*(temp_mem->host_mem.begin()), 0, NULL, nullptr);
                            // clFinish(queues_["buffer_rw"]);
                            //checkError(status, "Failed to read buffer from result");
                            // if(net_[processing->net_idx_].name == "csm_32"){
                            //     for(auto elem:temp_mem->host_mem){
                            //         std::cout << int(elem) << std::endl;
                            //     }
                            // }
                            temp_mem->mem_type = ALL;
                            // std::cout << temp_mem->host_mem.size() << std::endl;
                            inputs.emplace_back(temp_mem->host_mem);
                        }
                        else{
                            // if(net_[processing->net_idx_].name == "csm_32"){
                            //     for(auto elem:temp_mem->host_mem){
                            //         std::cout << int(elem) << std::endl;
                            //     }
                            // }
                            inputs.emplace_back(temp_mem->host_mem);
                        }
                    }
                    else {
                        for(auto& e_mem:lock_rmems){
                            e_mem.second->rw_mutex.unlock_shared();
                        }
                        LOG_DEBUG("input not ready:%s!\n", input.input_name.c_str());
                        return false;
                    }
                }
                else{
                    for(auto& e_mem:lock_rmems){
                        e_mem.second->rw_mutex.unlock_shared();
                    }
                    LOG_DEBUG("no invalid input:%s!\n", input.input_name.c_str());
                    return false;
                }
            }
        }
        std::vector<std::reference_wrapper<std::vector<T, AlignedAllocator<T, 64>>>> outputs;
        for(auto& output:net_[processing->net_idx_].outputs){
            auto it = processing->Mem_map_.find(output.output_name);
            if(it != processing->Mem_map_.end()) {
                auto it = lock_rmems.find(output.output_name);
                if (it != lock_rmems.end()) {
                    lock_rmems[output.output_name]->rw_mutex.unlock_shared();
                    lock_rmems.erase(it);
                }
                std::shared_ptr<Mem<std::vector<T, AlignedAllocator<T, 64>>>> temp_mem = processing->Mem_map_[output.output_name];
                if(temp_mem->rw_mutex.try_lock()){
                    lock_wmems[output.output_name] = temp_mem;
                    if((temp_mem->mem_type == DEVICE)  | (temp_mem->mem_type == ALL)){
                        clReleaseMemObject(temp_mem->device_mem);
                        temp_mem->device_mem = nullptr;
                    }
                    temp_mem->mem_type = HOST;
                    outputs.emplace_back(temp_mem->host_mem); 
                }
                else{
                    for(auto& e_mem:lock_rmems){
                        e_mem.second->rw_mutex.unlock_shared();
                    }
                    for(auto& e_mem:lock_wmems){
                        e_mem.second->rw_mutex.unlock();
                    }
                    LOG_DEBUG("output not ready:%s!\n", output.output_name.c_str());
                    return false;
                }
            }
            else{
                auto temp_mem = std::make_shared<Mem<std::vector<T, AlignedAllocator<T, 64>>>>(HOST, std::move(std::vector<T, AlignedAllocator<T, 64>>(output.output_size)), output.output_size);
                processing->Mem_map_[output.output_name] = temp_mem;
                temp_mem->rw_mutex.lock();
                outputs.emplace_back(temp_mem->host_mem);
            }
        }
        for(int i=0; i < thread_num-1; i++){
            auto task = [thread_num, i, func, inputs, outputs]{
                func(thread_num, i, inputs, outputs);
            };
            std::packaged_task<void()> pt(task);
            std::future<void> temp_future = pt.get_future();
            host_task_queue_.push(std::move(pt));
            futures.emplace_back(std::move(temp_future));
        }
        auto task = [this, processing, thread_num, func, inputs, outputs, real_futures]{
            func(thread_num, thread_num - 1, inputs, outputs);
            for(auto& future:*real_futures){
                future.wait();
            }
            donext_queue_.push(processing);
        };
        std::packaged_task<void()> pt(std::move(task));
        host_task_queue_.push(std::move(pt));
        LOG_DEBUG("start_host_function %s done!!!", func_name.c_str());
        return true;
    }
    else{
        Last_function func = last_function_;
        std::vector<std::reference_wrapper<std::vector<T, AlignedAllocator<T, 64>>>> inputs;
        for(auto& input:net_[processing->net_idx_].inputs){
            auto it = processing->Mem_map_.find(input.input_name);
            if(it != processing->Mem_map_.end()) {
                std::shared_ptr<Mem<std::vector<T, AlignedAllocator<T, 64>>>> temp_mem = processing->Mem_map_[input.input_name];
                if(temp_mem->rw_mutex.try_lock_shared()){
                    lock_rmems[input.input_name] = temp_mem;
                    if(temp_mem->mem_type == DEVICE){
                        cl_int status;
                        temp_mem->host_mem.resize(input.input_size);
                        status = clEnqueueReadBuffer(queues_["buffer_rw"], temp_mem->device_mem, CL_TRUE, 0, sizeof(T) * input.input_size, &*(temp_mem->host_mem.begin()), 0, NULL, nullptr);
                        // clFinish(queues_["buffer_rw"]);
                        //checkError(status, "Failed to read buffer from result");
                        temp_mem->mem_type = ALL;
                        inputs.emplace_back(temp_mem->host_mem);
                    }
                    else{
                        // for(auto elem:temp_mem->host_mem){
                        //     std::cout << int(elem) << std::endl;
                        // }
                        inputs.emplace_back(temp_mem->host_mem);
                    }
                }
                else{
                    for(auto& e_mem:lock_rmems){
                        e_mem.second->rw_mutex.unlock_shared();
                    }
                    LOG_DEBUG("last function input not ready:%s!\n", input.input_name.c_str());
                    return false;
                }
            }
            else{
                for(auto& e_mem:lock_rmems){
                    e_mem.second->rw_mutex.unlock_shared();
                }
                LOG_DEBUG("no invalid input:%s!\n", input.input_name.c_str());
                return false;
            }
        }
        auto& output = net_[processing->net_idx_].outputs[0];
        processing->m_outputs_ = std::make_shared<Mem<std::vector<last_function_output_type>>>(HOST, std::move(std::vector<last_function_output_type>(output.output_size)), output.output_size);
        processing->m_outputs_->rw_mutex.lock();
        auto task = [this, processing, func, inputs]{
            func(inputs, processing->m_outputs_->host_mem);
            donext_queue_.push(processing);
            LOG_DEBUG("last function done!\n");
        };
        std::packaged_task<void()> pt(task);
        host_task_queue_.push(std::move(pt));
        LOG_DEBUG("start_host_function  done!!!");
        return true;
    }
}

template<typename T, typename Last_function>
void NetRun<T, Last_function>::start_kernel(std::shared_ptr<ProcessingInfo<T, Last_function>> processing, std::list<std::shared_ptr<ProcessingInfo<T, Last_function>>>& donext_list){
    std::string kernel_name = net_[processing->net_idx_].name;
    cl_int status;
    cl_mem temp_in;
    cl_mem temp_out;
    bool flag;
    unsigned arg_index = 0;
    std::unordered_map<std::string, std::shared_ptr<Mem<std::vector<T, AlignedAllocator<T, 64>>>>> lock_rmems;
    std::unordered_map<std::string, std::shared_ptr<Mem<std::vector<T, AlignedAllocator<T, 64>>>>> lock_wmems;
    for(const auto &input:net_[processing->net_idx_].inputs){
        if(input.input_name.substr(0, 7) == "weights"){
            std::shared_ptr<Mem<std::vector<T, AlignedAllocator<T, 64>>>> temp_mem = net_weights_[input.input_name];
            if(temp_mem->mem_type == HOST){
                temp_in = clCreateBuffer(context_, CL_MEM_READ_ONLY, input.input_size * sizeof(T), nullptr, &status);
                temp_mem->mem_type = ALL;
                temp_mem->device_mem = temp_in;
                //checkError(status, "Failed to create buffer for Layer%d input", 0);
                clEnqueueWriteBuffer(queues_["buffer_rw"], temp_in, CL_TRUE, 0, input.input_size * sizeof(T), &*((temp_mem->host_mem).begin()), 0, nullptr, nullptr);
                clFinish(queues_["buffer_rw"]);
                // processing->result_map_[input.input_name] =  std::make_shared<Mem<T>>(DEVICE, temp_in);
            }
            else{
                temp_in = temp_mem->device_mem;
            }
            add_arg(arg_index, kernel_name, temp_in);
        }
        else{
            auto it = processing->Mem_map_.find(input.input_name);
            if (it != processing->Mem_map_.end()) {
                if(it->second->rw_mutex.try_lock_shared()){
                    LOG_DEBUG("lock_shared:%s -> at layer:%u", input.input_name.c_str(), processing->net_idx_);
                    lock_rmems[input.input_name] = it->second;
                    if(it->second->mem_type == HOST){
                        temp_in = clCreateBuffer(context_, CL_MEM_READ_WRITE, input.input_size * sizeof(T), nullptr, &status);
                        it->second->mem_type = ALL;
                        it->second->device_mem = temp_in;
                        //checkError(status, "Failed to create buffer for Layer%d input", 0);
                        clEnqueueWriteBuffer(queues_["buffer_rw"], temp_in, CL_TRUE, 0, input.input_size * sizeof(T),&*((it->second->host_mem).begin()), 0, nullptr, nullptr);
                        clFinish(queues_["buffer_rw"]);
                        // processing->result_map_[input.input_name] =  std::make_shared<Mem<T>>(DEVICE, temp_in);
                    }
                    else{
                        temp_in = it->second->device_mem;
                    }
                }
                else{
                    for(auto& e_mem:lock_rmems){
                        e_mem.second->rw_mutex.unlock_shared();
                    }
                    LOG_DEBUG("input not ready:%s!\n", input.input_name.c_str());
                    return;
                }
                // processing->result_map_.erase(it);
            } else {
                for(auto& e_mem:lock_rmems){
                    e_mem.second->rw_mutex.unlock_shared();
                }
                LOG_DEBUG("start_kernel failed, input not ok!");
                return;
            }
            add_arg(arg_index, kernel_name, temp_in);
        }
    }
    for(const auto &output:net_[processing->net_idx_].outputs){
        if(processing->net_idx_ < net_.size() - 1){
            auto it = processing->Mem_map_.find(output.output_name);
            if(it != processing->Mem_map_.end()) {
                auto it = lock_rmems.find(output.output_name);
                if (it != lock_rmems.end()) {
                    lock_rmems[output.output_name]->rw_mutex.unlock_shared();
                    lock_rmems.erase(it);
                }
                std::shared_ptr<Mem<std::vector<T, AlignedAllocator<T, 64>>>> temp_mem = processing->Mem_map_[output.output_name];
                if(temp_mem->rw_mutex.try_lock()){
                    LOG_DEBUG("lock:%s -> at layer:%u", output.output_name.c_str(), processing->net_idx_);
                    lock_wmems[output.output_name] = temp_mem;
                    if((temp_mem->mem_type == HOST)  | (temp_mem->mem_type == ALL)){
                        temp_mem->host_mem.clear();
                    }
                    temp_mem->mem_type = DEVICE;
                    temp_out = temp_mem->device_mem;
                }
                else{
                    for(auto& e_mem:lock_rmems){
                        e_mem.second->rw_mutex.unlock_shared();
                    }
                    for(auto& e_mem:lock_wmems){
                        e_mem.second->rw_mutex.unlock();
                    }
                    LOG_DEBUG("output not ready:%s!\n", output.output_name.c_str());
                    return;
                }
            }
            else{
                temp_out = clCreateBuffer(context_, CL_MEM_READ_WRITE, output.output_size * sizeof(T), nullptr, &status);
                //checkError(status, "Failed to create buffer for output");
                processing->Mem_map_[output.output_name] = std::make_shared<Mem<std::vector<T, AlignedAllocator<T, 64>>>>(DEVICE, temp_out, output.output_size);
                processing->Mem_map_[output.output_name]->rw_mutex.lock();
                LOG_DEBUG("lock:%s -> at layer:%u", output.output_name.c_str(), processing->net_idx_);
            }
        }
        else{
            temp_out = clCreateBuffer(context_, CL_MEM_READ_WRITE, output.output_size * sizeof(last_function_output_type), nullptr, &status);
            //checkError(status, "Failed to create buffer for output");
            processing->m_outputs_ = std::make_shared<Mem<std::vector<last_function_output_type>>>(DEVICE, temp_out, output.output_size);
            processing->m_outputs_->rw_mutex.lock();
            flag = true;
        }
        add_arg(arg_index, kernel_name, temp_out);
    }
    LOG_DEBUG("clEnqueueTask:%s\n", kernel_name.c_str());
    status = clEnqueueTask(queues_[kernel_name], kernels_[kernel_name], 0, nullptr, &events_[kernel_name]);
    checkError(status, "Failed to launch kernel %s kernel", kernel_name);
    LOG_DEBUG("clEnqueueTask:%s, done\n", kernel_name.c_str());
    if(flag){
        processing->m_outputs_events_.push_back(events_[kernel_name]);
    }
    for(const auto &output:net_[processing->net_idx_].outputs){
        if(output.output_name != "output"){
            LOG_DEBUG("add %s event", output.output_name.c_str());
            processing->Event_map_[output.output_name] = events_[kernel_name];
        }
    }
    //checkError(status, "Failed to start kernel:%s", kernel_name);
    LOG_DEBUG("start kernel:%s success", kernel_name.c_str());
    device_request_idx_[kernel_name] = processing->net_idx_;
    donext_list.push_back(processing);
    return;
}

template<typename T, typename Last_function>
template<typename Arg>
void NetRun<T, Last_function>::add_arg(unsigned &arg_index, const std::string& kernel_name, Arg& kernel_arg){
    cl_int status = clSetKernelArg(kernels_[kernel_name], arg_index++, sizeof(Arg *), &kernel_arg);
    //checkError(status, "Failed to set %uth arg of kernel %s", arg_index, kernel_name.c_str());
    // start_kernel(arg_index, kernel_name, args...);
}

template<typename T, typename Last_function>
NetRun<T, Last_function>::NetRun(const Net& net, const PreNet& pre_net, const std::unordered_map<std::string, std::vector<T, AlignedAllocator<T, 64>>>& net_weights, const std::unordered_map<std::string, Host_function>& functions, Last_function last_function, const std::vector<cl_program>& program, const std::vector<cl_device_id>& device, const cl_context& context, int thread_num)
:net_(net), pre_net_(pre_net), functions_(functions), last_function_(last_function), program_(program), device_(device), context_(context){
    if(thread_num > 1){
        thread_num_ = thread_num;
    }
    else{
        thread_num_ = std::thread::hardware_concurrency();
        // std::cout << thread_num_ << std::endl;
    }
    thread_num_ = thread_num_>1?thread_num_:2;
    cl_int status;
    for(auto it:net_weights){
        net_weights_[it.first] = std::make_shared<Mem<std::vector<T, AlignedAllocator<T, 64>>>>(HOST, it.second, it.second.size());
    }
    for(auto &task:pre_net_){
        if(task.device_type == DEVICE){
            cl_mem temp_in;
            unsigned arg_index = 0;
            std::string kernel_name = task.name;
            int device_index = task.device_index; 
            kernels_[kernel_name] = clCreateKernel(program_[device_index], kernel_name.c_str(), &status);
            //checkError(status, "Failed to create kernel:%s", kernel_name.c_str());
            queues_[kernel_name] = clCreateCommandQueue(context_, device_[device_index], CL_QUEUE_PROFILING_ENABLE, &status);
            //checkError(status, "Failed to create command:%s", kernel_name);
            events_[kernel_name] = NULL;
            for(const auto &input:task.inputs){
                temp_in = clCreateBuffer(context_, CL_MEM_READ_ONLY, input.input_size * sizeof(T), nullptr, &status);
                //checkError(status, "Failed to create buffer for Layer%d input", 0);
                clEnqueueWriteBuffer(queues_["buffer_rw"], temp_in, CL_TRUE, 0, input.input_size * sizeof(T), &*(net_weights_[input.input_name]->host_mem.begin()), 0, nullptr, nullptr);
                // clFinish(queues_["buffer_rw"]);
                add_arg(arg_index, kernel_name, temp_in);
            }
            cl_int status = clEnqueueTask(queues_[kernel_name], kernels_[kernel_name], 0, nullptr, &events_[kernel_name]);
            //checkError(status, "Failed to start pre kernel:%s", kernel_name);
            LOG_DEBUG("start pre kernel:%s success", kernel_name.c_str());
        }
    }
    for(size_t i=0;i<net_.size();i++){
        auto& task = net_[i];
        if(task.device_type == DEVICE){
            std::string kernel_name = task.name;
            int device_index = task.device_index; 
            kernels_[kernel_name] = clCreateKernel(program_[device_index], kernel_name.c_str(), &status);
            checkError(status, "Failed to create kernel:%s", kernel_name.c_str());
            queues_[kernel_name] = clCreateCommandQueue(context_, device_[device_index], CL_QUEUE_PROFILING_ENABLE, &status);
            checkError(status, "Failed to create command:%s, on device:%d", kernel_name, device_index);
            events_[kernel_name] = NULL;
            for(auto &input:task.inputs){
                kernels_inputs_[kernel_name].push_back(input.input_name);
            }
            for(auto &output:task.outputs){
                kernels_outputs_[kernel_name].push_back(output.output_name);
            }
        }
        for(auto &input:task.inputs){
            mem_life_end_[input.input_name] = i;
        }
        for(auto &output:task.outputs){
            mem_life_end_[output.output_name] = i;
        }
    }
    // LOG_DEBUG("final_layers num:%ld", final_layers_);
    // for(int i=0;i<device_.size();i++){
    //     queues_["buffer_rw_"+ std::to_string(i)] = clCreateCommandQueue(context_, device_[i], CL_QUEUE_PROFILING_ENABLE, &status);
    // }
    queues_["buffer_rw"] = clCreateCommandQueue(context_, device_[0], CL_QUEUE_PROFILING_ENABLE, &status);
    //checkError(status, "Failed to create command:%s", "buffer_rw");
}

template<typename T, typename Last_function>
NetRun<T, Last_function>::~NetRun(){
    LOG_DEBUG("~NetRun!");
    stop();
    for(auto & thread:thread_pool_){
        thread.join();
    }
}

template<typename T, typename Last_function>
void NetRun<T, Last_function>::appendImage(const std::vector<T,  AlignedAllocator<T, 64>>& image){
    image_queue_.push(image);
}

template<typename T, typename Last_function>
std::vector<typename NetRun<T, Last_function>::last_function_output_type> NetRun<T, Last_function>::getResult(){
    std::shared_ptr<ProcessingInfo<T, Last_function>> processing = std::move(*(final_request_queue_.wait_and_pop()));
    std::vector<last_function_output_type> result = std::move(processing->getFinalResult());
    popProcessing();
    return result;
}

// template<typename T, typename Last_function>
// bool NetRun<T, Last_function>::getResult(std::vector<std::vector<T,  AlignedAllocator<T, 64>>>& res){
//     if(result_queue_len_.load(std::memory_order_relaxed) != 0){
//         result_queue_locker_.lock();
//         res = std::move(result_queue_.front());
//         result_queue_.pop_front();
//         result_queue_locker_.unlock();
//         result_queue_len_.fetch_sub(1,std::memory_order_relaxed);
//         return true;
//     }
//     else{
//         return false;
//     }
// }

template<typename T, typename Last_function>
void NetRun<T, Last_function>::popProcessing(){
    processinging_queue_.pop_front();
}

template<typename T, typename Last_function>
void NetRun<T, Last_function>::calculate(){
    while(!stop_){
        std::packaged_task<void()> pt;
        if(host_task_queue_.try_pop(pt)){
            pt();
        }
        else{
            std::this_thread::yield();
        }
    }
}

template<typename T, typename Last_function>
void NetRun<T, Last_function>::listen(){
    while(!stop_){
        std::vector<T,  AlignedAllocator<T, 64>> image_data;
        if(image_queue_.try_pop(image_data)){
            LOG_DEBUG("New Image come!");
            std::shared_ptr<ProcessingInfo<T, Last_function>> temp = std::make_shared<ProcessingInfo<T, Last_function>>(this,std::move(image_data));
            temp->commitRequest();
            processinging_queue_.push_back(temp);
        }
        std::list<std::shared_ptr<ProcessingInfo<T, Last_function>>> donext_list;
        auto device_request_ptr = device_request_queue_.begin();
        while(device_request_ptr != device_request_queue_.end()){
            // LOG_DEBUG("New loop!!!!!!!!!!!!!");
            std::string kernel_name = device_request_ptr->first;
            // LOG_DEBUG("get request at kernel:%s", kernel_name.c_str());
            if(events_[kernel_name] != NULL){
                cl_int status;
                LOG_DEBUG("clGetEventInfo!");
                cl_int err = clGetEventInfo(events_[kernel_name], CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(status), &status, NULL);
                if (err != CL_SUCCESS) {
                    printf("Error: Failed to get %s event info. Error code: %d\n", kernel_name.c_str(), err);
                    return;
                }
                LOG_DEBUG("clGetEventInfo  done!!!!!!!!!");
                if(status == CL_COMPLETE){
                    LOG_DEBUG("%s:COMPLETE!", kernel_name.c_str());
                    std::shared_ptr<ProcessingInfo<T, Last_function>> processing = device_request_ptr->second.front();
                    std::unordered_set<std::string> output_names;
                    for(const auto &output_name:kernels_outputs_[kernel_name]){
                        LOG_DEBUG("unlock:%s -> at layer:%u", output_name.c_str(), device_request_idx_[kernel_name]);
                        processing->Mem_map_[output_name]->rw_mutex.unlock();
                        output_names.insert(output_name);
                        if(device_request_idx_[kernel_name] >= mem_life_end_[output_name]){
                            LOG_DEBUG("kill mem:%s -> at layer:%u > %u", output_name.c_str(), device_request_idx_[kernel_name], mem_life_end_[output_name]);
                            processing->Mem_map_[output_name]->rw_mutex.kill();
                        }
                        if(processing->Mem_map_[output_name]->rw_mutex.iskilled() && processing->Mem_map_[output_name]->rw_mutex.isfree()){
                            LOG_DEBUG("kill mem:%s -> at layer:%u > %u", output_name.c_str(), device_request_idx_[kernel_name], mem_life_end_[output_name]);
                            processing->Mem_map_.erase(output_name);
                        }
                    }
                    for(const auto &input_name:kernels_inputs_[kernel_name]){
                        if(input_name.substr(0, 7) != "weights"){
                            auto it = output_names.find(input_name);
                            if(it == output_names.end()){
                                LOG_DEBUG("unlock_shared:%s -> at layer:%u", input_name.c_str(), device_request_idx_[kernel_name]);
                                processing->Mem_map_[input_name]->rw_mutex.unlock_shared();
                                if(device_request_idx_[kernel_name] >= mem_life_end_[input_name]){
                                    LOG_DEBUG("kill mem:%s -> at layer:%u > %u", input_name.c_str(), device_request_idx_[kernel_name], mem_life_end_[input_name]);
                                    processing->Mem_map_[input_name]->rw_mutex.kill();
                                }
                                if(processing->Mem_map_[input_name]->rw_mutex.iskilled() && processing->Mem_map_[input_name]->rw_mutex.isfree()){
                                    LOG_DEBUG("kill mem:%s -> at layer:%u > %u", input_name.c_str(), device_request_idx_[kernel_name], mem_life_end_[input_name]);
                                    processing->Mem_map_.erase(input_name);
                                }
                            }
                        }
                    }
                    device_request_ptr->second.pop_front();
                    clReleaseEvent(events_[kernel_name]);
                    events_[kernel_name] = NULL;
                    if(device_request_ptr->second.size()>0){
                        processing = device_request_ptr->second.front();
                        start_kernel(processing, donext_list);
                    }
                    else{
                        device_request_queue_.erase(device_request_ptr++);
                        continue;
                    }
                }
                else{
                    LOG_DEBUG("%s's status:%d, %p\n", kernel_name.c_str(), status,device_request_ptr->second.front().get());
                    // LOG_DEBUG("%s status:%d", status);
                    // LOG_DEBUG("CL_QUEUED:%d", CL_QUEUED);
                    // LOG_DEBUG("CL_SUBMITTED:%d", CL_SUBMITTED);
                    // LOG_DEBUG("CL_RUNNING:%d", CL_RUNNING);
                    // LOG_DEBUG("CL_COMPLETE:%d", CL_COMPLETE);
                } 
            }
            else{
                std::shared_ptr<ProcessingInfo<T, Last_function>> processing = device_request_ptr->second.front();
                start_kernel(processing, donext_list);
                // device_request_idx_[kernel_name] = processing->net_idx_;
            }
            ++device_request_ptr;
        }
        for(auto& elem:donext_list){
            elem->doNext();
        }
        // LOG_DEBUG("host_request_ptr!!!!!!!!");
        auto host_request_ptr = host_request_queue_.begin();
        while(host_request_ptr != host_request_queue_.end()){
            std::shared_ptr<ProcessingInfo<T, Last_function>> processing = *host_request_ptr;
            bool success = start_host_function(processing);
            if(success){
                host_request_queue_.erase(host_request_ptr++);
                continue;
            }
            ++host_request_ptr;
        }
        std::shared_ptr<ProcessingInfo<T, Last_function>> donext_data;
        while(donext_queue_.try_pop(donext_data)){
            LOG_DEBUG("wwwwwwwwwwww!");
            std::shared_ptr<ProcessingInfo<T, Last_function>> processing = donext_data;
            std::unordered_set<std::string> output_names;
            for(const auto &output:net_[processing->net_idx_].outputs){
                if(processing->net_idx_ < net_.size() - 1){
                    processing->Mem_map_[output.output_name]->rw_mutex.unlock();
                }
                else{
                    processing->m_outputs_->rw_mutex.unlock();
                }
                output_names.insert(output.output_name);
            }
            LOG_DEBUG("aaaaaaa!");
            for(const auto &input:net_[processing->net_idx_].inputs){
                if(input.input_name.substr(0, 7) != "weights"){
                    auto it = output_names.find(input.input_name);
                    if(it == output_names.end()){
                        processing->Mem_map_[input.input_name]->rw_mutex.unlock_shared();
                    }
                    if(processing->net_idx_ >= mem_life_end_[input.input_name]){
                        LOG_DEBUG("erase mem:%s -> at layer:%lu > %u", input.input_name.c_str(), processing->net_idx_, mem_life_end_[input.input_name]);
                        processing->Mem_map_.erase(input.input_name);
                    }
                }
            }
            LOG_DEBUG("bbbbbb!");
            processing->doNext();
        }
        std::this_thread::yield();
        // LOG_DEBUG("final_request_ptr!!!!!!!!");
        // auto final_request_ptr = final_request_queue_.begin();
        // while(final_request_ptr != final_request_queue_.end()){
        //     if((*final_request_ptr)->outputs_finished_ == final_layers_){
        //         LOG_DEBUG("final task done, outputs_finished_:%ld!", (*final_request_ptr)->outputs_finished_);
        //         (*final_request_ptr)->getFinalResult();
        //         final_request_queue_.erase(final_request_ptr++);
        //         continue;
        //     }
        //     else{
        //         LOG_DEBUG("outputs_finished:%ld", (*final_request_ptr)->outputs_finished_);
        //     }
        //     ++final_request_ptr;
        // }
    }
}

template<typename T, typename Last_function>
void NetRun<T, Last_function>::run(){
    stop_ = false;
    thread_pool_.push_back(std::thread([this]() { this->listen(); }));
    for(unsigned i=1;i<thread_num_;++i){
        thread_pool_.push_back(std::thread([this]() { this->calculate(); }));
    }
}

template<typename T, typename Last_function>
void NetRun<T, Last_function>::stop(){
    LOG_DEBUG("stop!");
    stop_ = true;
}

#endif