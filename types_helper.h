

#pragma  once
#include <string>
#include "type_list.h"

template< typename...P>
struct Extract_Type_String;

template<>
struct Extract_Type_String<>
{
	static inline const std::string value = "";
};

template<typename H, typename ...P>
struct Extract_Type_String<H, P...>
{
	static inline const std::string  local_value = typeid(H).name();
	static inline const std::string  value = local_value + Extract_Type_String< P...>::value;
};


template<typename tl>
struct Extract_Name_Of_Type_List;

template<>
struct Extract_Name_Of_Type_List<type_list<>>
{
	static inline const std::string value = "";

};

template<typename H, typename ...P>
struct Extract_Name_Of_Type_List<type_list<H, P...>>
{
	static inline const std::string local_value = typeid(H).name();
	static inline const std::string value = local_value + Extract_Name_Of_Type_List<type_list<P...>>::value;
};

