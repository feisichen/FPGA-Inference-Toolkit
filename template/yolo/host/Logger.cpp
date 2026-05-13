//
// Created by fsc on 24-7-6.
//
#include "Logger.h"
#include <iostream>
#include <string>
#include <chrono>
#include <iomanip>
#include <ctime>
Logger& Logger::instance(){
    static Logger logger;
    return logger;
}

void Logger::setLogLevel(LogLevel level) {
    logLevel_ = level;
}

void Logger::log(std::string msg) {
    // 获取当前时间
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration).count();
    int milliseconds = millis - seconds * 1000;

    // 获取当前时间的秒部分
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm *tm = std::localtime(&now_time);

    // 打印时间戳，格式：HH:MM:SS.mmm
    std::cout << std::put_time(tm, "%H:%M:%S") << "." << std::setw(3) << std::setfill('0') << milliseconds; // 时:分:秒.毫秒
    std::cout << " " << Level_Info_[logLevel_] << ": " << msg << std::endl;
}
