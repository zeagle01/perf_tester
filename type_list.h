
#pragma once


template <typename ...P> 
struct type_list;


template <typename tl>
struct front_imp;

template <typename H,typename ...P>
struct front_imp<type_list<H, P...>>
{
	using type = H;
};

template<typename tl>
using front_t = typename front_imp<tl>::type;





