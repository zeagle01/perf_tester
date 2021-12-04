

#pragma once

#include "class_reflection.h"


template<typename Seq >
struct For_Each_In_Range_Imp;

template<size_t ... N>
struct For_Each_In_Range_Imp<std::index_sequence< N...>>
{
	template<typename F, typename ...P>
	static void apply(P&&...p)
	{
		((typename F:: template apply<N>(std::forward<P>(p)...)), ...);
	}
};

template<size_t N>
using  For_Each_In_Range = For_Each_In_Range_Imp<std::make_index_sequence<N>>;

template<typename Ta, typename Tb>
struct Is_Same_template { constexpr static bool value = false; };

template<typename A, template<typename> typename Ta>
struct Is_Same_template<Ta<A>, Ta<void>> { constexpr static bool value = true; };


template<typename F>
struct Dev_Ptr_Only
{
	template<size_t I, typename Tp0, typename ...Tps >
	static void apply(Tp0 tp0, Tps ...tps)
	{
		using element_type = std::remove_const_t<std::remove_reference_t<decltype(std::get<I>(tp0))>>;

		if constexpr (Is_Same_template<element_type, Device_Pointer<void>>::value)
		{
			F::apply(std::get<I>(tp0), std::get<I>(tps)...);
		}
	}
};

template<typename F>
struct Value_Only
{
	template<size_t I, typename Tp0, typename ...Tps >
	static void apply(Tp0 tp0, const Tps ...tps)
	{
		using element_type = std::remove_const_t<std::remove_reference_t<decltype(std::get<I>(tp0))>>;

		if constexpr (!Is_Same_template<element_type, Device_Pointer<void>>::value)
		{
			F::apply(std::get<I>(tp0), std::get<I>(tps)...);
		}
	}
};

template<typename T>
struct For_Each_Member
{
	template<template<typename> typename Filter, typename F, typename T0, typename ... Ts >
	static void apply(T0& t0, const Ts& ...ts)
	{
		auto a_tuple = clumsy_lib::as_tuple(t0);

		using tuple_type = decltype(a_tuple);
		constexpr auto size = std::tuple_size_v<tuple_type>;

		For_Each_In_Range<size>::template apply<Filter<F>>(
			a_tuple,
			clumsy_lib::as_tuple(ts)...
			);
	}

};

