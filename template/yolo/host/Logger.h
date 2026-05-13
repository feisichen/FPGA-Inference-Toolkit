//
// Created by fsc on 24-7-6.
//

#ifndef MYMUDUO_LOGGER_H
#define MYMUDUO_LOGGER_H

#include <string>
#include <iostream>
#include "noncopyable.h"
//日志级别 INFO ERROR（错误，但不影响执行）FATAL（致命问题） DEBUG
#define LOG_INFO(logmsgFormat, ...) \
    do                              \
    {                               \
           Logger &logger = Logger::instance(); \
           logger.setLogLevel(INFO); \
           char buf[1024] = {'\0'}; \
           snprintf(buf, 1024, logmsgFormat, ##__VA_ARGS__); \
           logger.log(buf); \
    }while(0)

#define LOG_ERROR(logmsgFormat, ...) \
    do                              \
    {                               \
           Logger &logger = Logger::instance(); \
           logger.setLogLevel(ERROR); \
           char buf[1024] = {'\0'}; \
           snprintf(buf, 1024, logmsgFormat, ##__VA_ARGS__); \
           logger.log(buf); \
    }while(0)

#define LOG_FATAL(logmsgFormat, ...) \
    do                              \
    {                               \
           Logger &logger = Logger::instance(); \
           logger.setLogLevel(FATAL); \
           char buf[1024] = {'\0'}; \
           snprintf(buf, 1024, logmsgFormat, ##__VA_ARGS__); \
           logger.log(buf);          \
           exit(-1); \
    }while(0)

#ifdef HOSTDEBUG
#define LOG_DEBUG(logmsgFormat, ...) \
    do                              \
    {                               \
           Logger &logger = Logger::instance(); \
           logger.setLogLevel(DEBUG); \
           char buf[1024] = {'\0'}; \
           snprintf(buf, 1024, logmsgFormat, ##__VA_ARGS__); \
           logger.log(buf); \
    }while(0)
#else
#define LOG_DEBUG(logmsgFormat, ...)
#endif
enum LogLevel{
    INFO,
    ERROR,
    FATAL,
    DEBUG
};
class Logger: noncopyable{
public:
    static Logger& instance();
    void setLogLevel(LogLevel level);
    void log(std::string msg);
private:
    const char * const Level_Info_[4] = {"[INFO]", "[ERROR]", "[FATAL]", "[DEBUG]"};
    int logLevel_;
    Logger() = default;
};
#endif //MYMUDUO_LOGGER_H
