#ifndef ALIGNEDALLOCATOR_H
#define ALIGNEDALLOCATOR_H

#include <cstddef>
#include <new>
#include <limits>

// 定义模板类 AlignedAllocator
template<typename ElementType, std::size_t ALIGNMENT_IN_BYTES = 64>
class AlignedAllocator
{
private:
    static_assert(
        ALIGNMENT_IN_BYTES >= alignof(ElementType),
        "Beware that types like int have minimum alignment requirements "
        "or access will result in crashes."
    );

public:
    // 类型定义，用于支持标准容器接口
    typedef ElementType value_type;

    // rebind 支持将分配器用于不同类型的元素
    template<class OtherElementType>
    struct rebind {
        typedef AlignedAllocator<OtherElementType, ALIGNMENT_IN_BYTES> other;
    };

public:
    // 默认构造函数
    AlignedAllocator() noexcept {}

    // 拷贝构造函数
    AlignedAllocator(const AlignedAllocator&) noexcept {}

    // 模板化的拷贝构造函数，用于处理不同类型的分配器之间的转换
    template<typename U>
    AlignedAllocator(const AlignedAllocator<U, ALIGNMENT_IN_BYTES>&) noexcept {}

    // 分配内存函数
    ElementType* allocate(std::size_t nElementsToAllocate) {
        if (nElementsToAllocate > std::numeric_limits<std::size_t>::max() / sizeof(ElementType)) {
            throw std::exception();
        }

        std::size_t nBytesToAllocate = nElementsToAllocate * sizeof(ElementType);

        // 使用 C++11 中的 aligned_alloc 替代 std::align_val_t
        void* ptr = nullptr;
        if (posix_memalign(&ptr, ALIGNMENT_IN_BYTES, nBytesToAllocate) != 0) {
            throw std::bad_alloc();
        }

        return reinterpret_cast<ElementType*>(ptr);
    }

    // 释放内存函数
    void deallocate(ElementType* allocatedPointer, std::size_t /* nBytesAllocated */) noexcept {
        // 直接使用 free() 来释放由 posix_memalign 分配的内存
        std::free(allocatedPointer);
    }

    // 判断两个分配器是否相等
    bool operator==(const AlignedAllocator<ElementType, ALIGNMENT_IN_BYTES>&) const noexcept {
        return true;  // 一般来说，同一类型的分配器总是相等的
    }

    bool operator!=(const AlignedAllocator<ElementType, ALIGNMENT_IN_BYTES>& other) const noexcept {
        return !(*this == other);  // 基于 operator== 的实现
    }
};

#endif // ALIGNEDALLOCATOR_H
