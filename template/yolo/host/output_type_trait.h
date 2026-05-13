#ifndef OUTPUT_TYPE_TRAIT_H_
#define OUTPUT_TYPE_TRAIT_H_

#include <vector>
#include "AlignedAllocator.h"
#include <functional>

template<typename T>
class output_type_trait;

// template<T, Output_Type>
// output_type_trait<typename void(*)(int, int, std::vector<std::reference_wrapper<std::vector<T, AlignedAllocator<T, 64>>>>, std::reference_wrapper<Output_Type>>){
// public:
// 	typedef  Output_Type output_type;
// }

template<typename T, typename Output_Type>
class output_type_trait<void(*)(int, int, std::vector<std::reference_wrapper<std::vector<T, AlignedAllocator<T, 64>>>>, std::vector<Output_Type> &)>{
public:
	typedef  Output_Type output_type;
};

template<typename T, typename Output_Type>
class output_type_trait<void(*)(std::vector<std::reference_wrapper<std::vector<T, AlignedAllocator<T, 64>>>>, std::vector<Output_Type> &)>{
public:
	typedef  Output_Type output_type;
};

#endif