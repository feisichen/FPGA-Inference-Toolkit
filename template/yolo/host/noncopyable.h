//
// Created by fsc on 24-7-6.
//

#ifndef MYMUDUO_NONCOPYABLE_H
#define MYMUDUO_NONCOPYABLE_H
#include <iostream>

class noncopyable{
public:
    noncopyable(const noncopyable&) = delete;
    noncopyable& operator=(const noncopyable&) = delete;
protected:
    noncopyable() = default;
    ~noncopyable() = default;
    noncopyable(noncopyable&&) noexcept = default;
    noncopyable& operator=(noncopyable&&) noexcept = default;
};

#endif //MYMUDUO_NONCOPYABLE_H
