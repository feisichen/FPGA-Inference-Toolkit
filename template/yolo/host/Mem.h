#ifndef MEM_H
#define MEM_H
#include "Layer.h"
#include <CL/opencl.h>
#include <thread>
#include <mutex>
#include "ocl_util.h"
#include "timer.h"
#include <vector>
#include <utility>
#include "Logger.h"
#include "AlignedAllocator.h"

template<typename T, typename Last_function>
class NetRun;

template<typename T, typename Last_function>
class ProcessingInfo;

template<typename T>
class Mem;

class RWLock{
public:
    void lock(){
        while(!try_lock()){
            std::this_thread::yield();
        }
    }
    bool try_lock(){
        std::unique_lock<std::mutex> lk(m_mutex);
        if(m_read_count == 0 && m_write_count == 0 ){
            m_write_count++;
            return true;
        }else if(m_write_count < 0 || m_read_count < 0){
            printf("what the hell??????????????????!!!!!!\n");
            return false;
        } else {
            return false;
        }
    }
    void unlock(){
        std::unique_lock<std::mutex> lk(m_mutex);
        m_write_count--;
    }
    void lock_shared(){
        while(!try_lock_shared()){
            std::this_thread::yield();
        }
    }
    bool try_lock_shared(){
        std::unique_lock<std::mutex> lk(m_mutex);
        if(m_write_count == 0){
            m_read_count++;
            return true;
        } else if(m_write_count < 0){
            printf("what the hell??????????????????!!!!!!\n");
            return false;
        }
        else{
            // printf("read:%d, write:%d\n", m_read_count, m_write_count);
            return false;
        }
    }
    void unlock_shared(){
        std::unique_lock<std::mutex> lk(m_mutex);
        m_read_count--;
    }
    bool isfree(){
        if(m_read_count == 0 && m_write_count == 0 ){
            return true;
        }
        else{
            return false;
        }
    }
    void kill(){
        m_killed = true;
    }
    bool iskilled(){
        return m_killed;
    }
private:
    std::mutex m_mutex;
    bool m_killed = false;
    int m_read_count = 0;
    int m_write_count = 0;
};

template<typename T>
class Mem{
public:
    Mem(){}
    Mem(DeviceType type, cl_mem mem, unsigned size):mem_type(type), device_mem(mem), m_size(size){}
    Mem(DeviceType type, const T& mem, unsigned size):mem_type(type), host_mem(mem), m_size(size){}
    Mem(DeviceType type, T&& mem, unsigned size):mem_type(type), host_mem(std::forward<T>(mem)), m_size(size){}
    Mem(const Mem<T>& other){
        mem_type = other.mem_type;
        device_mem = other.device_mem;
        host_mem = other.host_mem;
        m_size = other.m_size;
    }
    Mem(Mem<T>&& other) noexcept{
        DeviceType mem_type_temp = other.mem_type;
        cl_mem device_mem_temp = other.device_mem;
        T host_mem_temp = std::move(other.host_mem);
        unsigned m_size_temp = other.m_size;
        other.mem_type = NONE;
        other.device_mem = nullptr;
        other.m_size = 0;
        mem_type = mem_type_temp;
        device_mem = device_mem_temp;
        host_mem = std::move(host_mem_temp);
        m_size = m_size_temp;
    }
    ~Mem(){
        if(device_mem != nullptr){
            // std::cout << "release" << std::endl;
            clReleaseMemObject(device_mem);
        }
    }
public:
    RWLock rw_mutex;
    DeviceType mem_type;
    cl_mem device_mem = nullptr;
    T host_mem;
    unsigned m_size;
};
#endif
