#ifndef QUEUE_H
#define QUEUE_H

#include <queue>
#include <mutex>
#include <condition_variable>


//TODO
// template<typename T>
// class lockfree_queue{}

template<typename T>
class threadsafe_queue
{
private:
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::queue<std::shared_ptr<T>> queue_;
public:
    threadsafe_queue(){}
    threadsafe_queue(const std::queue<T>& other):queue_(other){}
    threadsafe_queue(std::queue<T>&& other) :queue_(std::move(other)) {}
    threadsafe_queue(const threadsafe_queue& other)
    {
        std::lock_guard<std::mutex> lk(other.mutex_);
        queue_ = other.queue_;
    }
    void push(const T& elem)
    {
        std::shared_ptr<T> data(std::make_shared<T>(elem));
        std::unique_lock<std::mutex> lk(mutex_);
        queue_.push(data);
        lk.unlock();
        condition_.notify_one();
    }
    void push(T&& elem)
    {
        std::shared_ptr<T> data(std::make_shared<T>(std::move(elem)));
        std::unique_lock<std::mutex> lk(mutex_);
        queue_.push(data);
        lk.unlock();
        condition_.notify_one();
    }
    void wait_and_pop(T& elem)
    {
        std::unique_lock<std::mutex> lk(mutex_);
        condition_.wait(lk, [this](){return !queue_.empty(); });
        elem = std::move(*(queue_.front()));
        queue_.pop();
    }
    std::shared_ptr<T> wait_and_pop()
    {
        std::unique_lock<std::mutex> lk(mutex_);
        condition_.wait(lk, [this](){return !queue_.empty(); });
        std::shared_ptr<T> data = queue_.front();
        queue_.pop();
        return data;
    }
    bool try_pop(T& elem)
    {
        std::lock_guard<std::mutex> lk(mutex_);
        if(queue_.empty())
        {
            return false;
        }
        elem = std::move(*(queue_.front()));
        queue_.pop();
        return true;
    }
    std::shared_ptr<T> try_pop()
    {
        std::lock_guard<std::mutex> lk(mutex_);
        if (queue_.empty())
        {
            return std::shared_ptr<T>();
        }
        std::shared_ptr<T> data = queue_.front();
        queue_.pop();
        return data;
    }
    bool empty()
    {
        std::lock_guard<std::mutex> lk(mutex_);
        return queue_.empty();
    }
};
#endif