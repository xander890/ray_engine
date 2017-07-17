// In DefineImprovedEnum.h

////////////////////////////////////////////////////////////////////////
// IMPORTANT NOTE:
// This is a "batch file of preprocessing directives"
// (because this cannot be done with a macro).
// Each time you include this file you are calling a batch file,
// it doesn't work as a macro include.
// If you want to declare several different enum types,
// you have to include this file several times.
// Do not use "#pragma once" directive, because it would have
// unexpected behaviour and results.
// Do not use directives like:
// #ifndef _IMPROVED_ENUM_H_ ; #define _IMPROVED_ENUM_H_ (same reason).
////////////////////////////////////////////////////////////////////////
// AUTHOR:      Hugo González Castro
// TITLE:       Improving C++ Enum: Adding Serialization,
//                                  Inheritance and Iteration.
// DESCRIPTION: A different approach to C++ enums: enum to string,
//              enum extension and enum iteration.
// VERSION:     v5.0 - 2009/04/13
// LICENSE:     CPOL (Code Project Open License).
//              Please, do not remove nor modify this header.
// URL:         ImprovedEnum.aspx
////////////////////////////////////////////////////////////////////////
// INPUT PARAMETERS:
// This file needs the following input parameters to be defined
// before including it:
// Input parameter: the name of the enumeration
// #define IMPROVED_ENUM_NAME [NameOfYourEnum]
// Input parameter: the file with the enum items
// #define IMPROVED_ENUM_FILE ["EnumItemFile"]
////////////////////////////////////////////////////////////////////////
// ENUMITEM FILE:
// The EnumItemFile is a list (one per line) of:
// ENUMITEM(EnumItem) or ENUMITEM_VALUE(EnumItem, Value)
////////////////////////////////////////////////////////////////////////
// ALTERNATIVE TO ENUMITEM FILE:
// IMPROVED_ENUM_LIST instead of IMPROVED_ENUM_FILE
// #define IMPROVED_ENUM_LIST  ENUMITEM(Item1) ... ENUMITEM(LastItem)
// #define IMPROVED_ENUM_LIST  ENUMITEM(Item1) \
//                             ENUMITEM(Item2) \
//                             ...
//                             ENUMITEM(LastItem)
////////////////////////////////////////////////////////////////////////
// OPTIONAL INPUT PARAMETERS:
// If you want to define a subclass instead of a namespace, you can
// #define IMPROVED_ENUM_SUBCLASS, or
// #define IMPROVED_ENUM_SUBCLASS_PARENT [ParentClass]
// to make subclass inherit from a ParentClass.
// If you want to extend an already defined ImprovedEnum, you have to
// define which type do you want to extend with
// IMPROVED_ENUM_INHERITED_NAME and IMPROVED_ENUM_INHERITED_FILE
// input parameters.
////////////////////////////////////////////////////////////////////////

// Checking ENUMITEM and ENUMITEM_VALUE macros are not already defined
#if defined(ENUMITEM)
#error ENUMITEM macro cannot be already defined
#elif defined(ENUMITEM_VALUE)
#error ENUMITEM_VALUE macro cannot be already defined
#endif

// Standard string class
#include <string>


#if defined(IMPROVED_ENUM_SUBCLASS_PARENT)

//! We define the IMPROVED_ENUM_NAME subclass (that
//! inherits from the specified parent class) which contains
//! the enum type and the static conversion methods from the
//! enum type to the string type and vice versa.
///////////////////////////////////////////////////////////
#define STATIC_METHOD static
class IMPROVED_ENUM_NAME : public IMPROVED_ENUM_SUBCLASS_PARENT
{
public:

#elif defined(IMPROVED_ENUM_SUBCLASS)

//! We define the IMPROVED_ENUM_NAME subclass, which contains
//! the enum type and the static conversion methods from the
//! enum type to the string type and vice versa.
///////////////////////////////////////////////////////////
#define STATIC_METHOD static
class IMPROVED_ENUM_NAME
{
public:

#else // IMPROVED_ENUM_SUBCLASS || IMPROVED_ENUM_SUBCLASS_PARENT

//! We define the IMPROVED_ENUM_NAME namespace, which contains
//! the enum type and the conversion functions from the
//! enum type to the string type and vice versa.
///////////////////////////////////////////////////////////
#define STATIC_METHOD
namespace IMPROVED_ENUM_NAME
{

#endif // IMPROVED_ENUM_SUBCLASS || IMPROVED_ENUM_SUBCLASS_PARENT

	//! Some stuff to get the string of the IMPROVED_ENUM_NAME
	///////////////////////////////////////////////////////////
#define GET_MACRO_STRING_EXPANDED(Macro)  #Macro
#define GET_MACRO_STRING(Macro)  GET_MACRO_STRING_EXPANDED(Macro)
#define ENUM_SEPARATOR  "::"
#define ENUM_TYPE_NAME  GET_MACRO_STRING(IMPROVED_ENUM_NAME)
	STATIC_METHOD inline const std::string EnumSeparator() { return ENUM_SEPARATOR; }
	STATIC_METHOD inline const std::string EnumTypeName() { return ENUM_TYPE_NAME; }
#ifdef  IMPROVED_ENUM_INHERITED_NAME
#define PARENT_ENUM_TYPE_NAME  GET_MACRO_STRING(IMPROVED_ENUM_INHERITED_NAME)
#define FULL_ENUM_TYPE_NAME    PARENT_ENUM_TYPE_NAME  ENUM_SEPARATOR  ENUM_TYPE_NAME
#else //IMPROVED_ENUM_INHERITED_NAME
#define PARENT_ENUM_TYPE_NAME  ""
#define FULL_ENUM_TYPE_NAME    ENUM_TYPE_NAME
#endif//IMPROVED_ENUM_INHERITED_NAME
	STATIC_METHOD inline const std::string ParentEnumTypeName()
	{ return PARENT_ENUM_TYPE_NAME; }
	STATIC_METHOD inline const std::string FullEnumTypeName()
	{ return FULL_ENUM_TYPE_NAME; }


	//! This defines the enumerated type:
	//////////////////////////////////////////
	typedef enum EnumTypeTag
	{
		//////////////////////////////////////////
		// With this mini-macro we make ENUMITEM file/s
		// a list of items separated by commas:
#define  ENUMITEM(EnumItem) EnumItem,
#define  ENUMITEM_VALUE(EnumItem, Value) EnumItem = Value,
#ifdef   IMPROVED_ENUM_INHERITED_FILE
#include IMPROVED_ENUM_INHERITED_FILE
#endif// IMPROVED_ENUM_INHERITED_FILE
#ifdef   IMPROVED_ENUM_FILE
#include IMPROVED_ENUM_FILE
#else // IMPROVED_ENUM_LIST
		IMPROVED_ENUM_LIST
#endif// IMPROVED_ENUM_FILE
#undef   ENUMITEM_VALUE
#undef   ENUMITEM
		//////////////////////////////////////////
		NotValidEnumItem // We add this item to all enums
	} EnumType, Type;

	//! Conversion from enum to string:
	//////////////////////////////////////////
	STATIC_METHOD inline const std::string Enum2String(const EnumType& t)
	{
		switch (t)
		{
			//////////////////////////////////////////
			// With this mini-macro we make ENUMITEM file/s
			// a CASE list which returns the stringized value:
#define  ENUMITEM(EnumItem) case EnumItem : return #EnumItem;
#define  ENUMITEM_VALUE(EnumItem, Value) ENUMITEM(EnumItem)
#ifdef   IMPROVED_ENUM_INHERITED_FILE
#include IMPROVED_ENUM_INHERITED_FILE
#endif// IMPROVED_ENUM_INHERITED_FILE
#ifdef   IMPROVED_ENUM_FILE
#include IMPROVED_ENUM_FILE
#else // IMPROVED_ENUM_LIST
			IMPROVED_ENUM_LIST
#endif// IMPROVED_ENUM_FILE
#undef   ENUMITEM_VALUE
#undef   ENUMITEM
				//////////////////////////////////////////
		}
		return ""; // NotValidEnumItem
	}

	//! Conversion from enum to full string (namespace::string):
	/////////////////////////////////////////////////////////////
	STATIC_METHOD inline const std::string Enum2FullString(const EnumType& t)
	{
		switch (t)
		{
			//////////////////////////////////////////
			// With this mini-macro we make ENUMITEM file/s
			// a CASE list which returns the stringized value:
#define  ENUMITEM(EnumItem) \
		case EnumItem : return  FULL_ENUM_TYPE_NAME  ENUM_SEPARATOR  #EnumItem;
#define  ENUMITEM_VALUE(EnumItem, Value) ENUMITEM(EnumItem)
#ifdef   IMPROVED_ENUM_INHERITED_FILE
#include IMPROVED_ENUM_INHERITED_FILE
#endif// IMPROVED_ENUM_INHERITED_FILE
#ifdef   IMPROVED_ENUM_FILE
#include IMPROVED_ENUM_FILE
#else // IMPROVED_ENUM_LIST
			IMPROVED_ENUM_LIST
#endif// IMPROVED_ENUM_FILE
#undef   ENUMITEM_VALUE
#undef   ENUMITEM
				//////////////////////////////////////////
		}
		return ""; // NotValidEnumItem
	}

	//! Conversion from string to enum:
	//////////////////////////////////////////
	STATIC_METHOD inline const EnumType String2Enum(const std::string& s)
	{
		if (s == "") return NotValidEnumItem;
		//////////////////////////////////////////
		// With this mini-macro we make ENUMITEM file/s
		// an IF list which returns the enum item:
#define  ENUMITEM(EnumItem) if (s == #EnumItem) return EnumItem;
#define  ENUMITEM_VALUE(EnumItem, Value) ENUMITEM(EnumItem)
#ifdef   IMPROVED_ENUM_INHERITED_FILE
#include IMPROVED_ENUM_INHERITED_FILE
#endif// IMPROVED_ENUM_INHERITED_FILE
#ifdef   IMPROVED_ENUM_FILE
#include IMPROVED_ENUM_FILE
#else // IMPROVED_ENUM_LIST
		IMPROVED_ENUM_LIST
#endif// IMPROVED_ENUM_FILE
#undef   ENUMITEM_VALUE
#undef   ENUMITEM
			//////////////////////////////////////////
			return NotValidEnumItem;
	}

	//! Conversion from full string (namespace::string) to enum:
	/////////////////////////////////////////////////////////////
	STATIC_METHOD inline const EnumType FullString2Enum(const std::string& s)
	{
		if (s == "") return NotValidEnumItem;
		//////////////////////////////////////////
		// With this mini-macro we make ENUMITEM file/s
		// an IF list which returns the enum item:
#define  ENUMITEM(EnumItem) \
	if (s ==  FULL_ENUM_TYPE_NAME  ENUM_SEPARATOR  #EnumItem) return EnumItem;
#define  ENUMITEM_VALUE(EnumItem, Value) ENUMITEM(EnumItem)
#ifdef   IMPROVED_ENUM_INHERITED_FILE
#include IMPROVED_ENUM_INHERITED_FILE
#endif// IMPROVED_ENUM_INHERITED_FILE
#ifdef   IMPROVED_ENUM_FILE
#include IMPROVED_ENUM_FILE
#else // IMPROVED_ENUM_LIST
		IMPROVED_ENUM_LIST
#endif// IMPROVED_ENUM_FILE
#undef   ENUMITEM_VALUE
#undef   ENUMITEM
			//////////////////////////////////////////
			return NotValidEnumItem;
	}

	//! Enum iteration to next:
	//////////////////////////////////////////
	STATIC_METHOD inline const EnumType NextEnumItem(const EnumType& t)
	{
		switch (t)
		{
		case NotValidEnumItem : 
			//////////////////////////////////////////
			// With this mini-macro we make ENUMITEM file/s
			// a CASE list which returns the next item:
#define  ENUMITEM(EnumItem) return EnumItem; case EnumItem : 
#define  ENUMITEM_VALUE(EnumItem, Value) ENUMITEM(EnumItem)
#ifdef   IMPROVED_ENUM_INHERITED_FILE
#include IMPROVED_ENUM_INHERITED_FILE
#endif// IMPROVED_ENUM_INHERITED_FILE
#ifdef   IMPROVED_ENUM_FILE
#include IMPROVED_ENUM_FILE
#else // IMPROVED_ENUM_LIST
			IMPROVED_ENUM_LIST
#endif// IMPROVED_ENUM_FILE
#undef   ENUMITEM_VALUE
#undef   ENUMITEM
				//////////////////////////////////////////
				return NotValidEnumItem; // (This indentation is intentional)
		}
		return NotValidEnumItem; // (This line is intentional too, do not remove)
	}

	//! Enum iteration to previous:
	//////////////////////////////////////////
	STATIC_METHOD inline const EnumType PreviousEnumItem(const EnumType& t)
	{
		EnumType tprev = NotValidEnumItem;
		//////////////////////////////////////////
		// With this mini-macro we make ENUMITEM file/s
		// an IF list which returns the previous item:
#define  ENUMITEM(EnumItem) \
	if (t == EnumItem) return tprev; else tprev = EnumItem;
#define  ENUMITEM_VALUE(EnumItem, Value) ENUMITEM(EnumItem)
#ifdef   IMPROVED_ENUM_INHERITED_FILE
#include IMPROVED_ENUM_INHERITED_FILE
#endif// IMPROVED_ENUM_INHERITED_FILE
#ifdef   IMPROVED_ENUM_FILE
#include IMPROVED_ENUM_FILE
#else // IMPROVED_ENUM_LIST
		IMPROVED_ENUM_LIST
#endif// IMPROVED_ENUM_FILE
#undef   ENUMITEM_VALUE
#undef   ENUMITEM
			//////////////////////////////////////////
			return tprev;
	}

	//! The first and the last Enums:
	//////////////////////////////////////////
	STATIC_METHOD inline const EnumType FirstEnumItem()
	{ return NextEnumItem(NotValidEnumItem); }
	STATIC_METHOD inline const EnumType LastEnumItem()
	{ return PreviousEnumItem(NotValidEnumItem); }
	STATIC_METHOD inline const std::string AllElementsString()
	{
		EnumType first = FirstEnumItem();
		std::string res = string("");
		do
		{
			res += string(" ") + Enum2String(first);
			first = NextEnumItem(first);
		} while(first != NotValidEnumItem);
		return res;
	}
	//! Number of enum items:
	//////////////////////////////////////////
	STATIC_METHOD inline const int NumberOfValidEnumItem()
	{
		return 0
			//////////////////////////////////////////
			// With this mini-macro we make ENUMITEM file/s
			// a counter list:
#define  ENUMITEM(EnumItem) +1
#define  ENUMITEM_VALUE(EnumItem, Value) ENUMITEM(EnumItem)
#ifdef   IMPROVED_ENUM_INHERITED_FILE
#include IMPROVED_ENUM_INHERITED_FILE
#endif// IMPROVED_ENUM_INHERITED_FILE
#ifdef   IMPROVED_ENUM_FILE
#include IMPROVED_ENUM_FILE
#else // IMPROVED_ENUM_LIST
			IMPROVED_ENUM_LIST
#endif// IMPROVED_ENUM_FILE
#undef   ENUMITEM_VALUE
#undef   ENUMITEM
			//////////////////////////////////////////
			;
	}

	// This is only needed when working with inherited/extended enums:
	////////////////////////////////////////////////////////////////////
#ifdef IMPROVED_ENUM_INHERITED_NAME
	//! Conversion from inherited enums:
	//! The same class items are returned without change, but
	//! other items are converted from one namespace to the other:
	//////////////////////////////////////////
	STATIC_METHOD inline const EnumType Inherited2Enum(const EnumType& t)
	{ return t; }
	STATIC_METHOD inline const EnumType Inherited2Enum(
		const IMPROVED_ENUM_INHERITED_NAME::EnumType& t)
	{
		switch (t)
		{
			//////////////////////////////////////////
			// With this mini-macro we make ENUMITEM file
			// a CASE list which returns the converted value
			// from one namespace to the other:
#define  ENUMITEM(EnumItem) \
		case IMPROVED_ENUM_INHERITED_NAME::EnumItem : return EnumItem;
#define  ENUMITEM_VALUE(EnumItem, Value) ENUMITEM(EnumItem)
#ifdef   IMPROVED_ENUM_INHERITED_FILE
#include IMPROVED_ENUM_INHERITED_FILE
#endif// IMPROVED_ENUM_INHERITED_FILE
#undef   ENUMITEM_VALUE
#undef   ENUMITEM
			//////////////////////////////////////////
		}
		return NotValidEnumItem;
	}

	//! Conversion to inherited enums:
	//! The same class items are returned without change, but
	//! other items are converted from one namespace to the other:
	//////////////////////////////////////////
	STATIC_METHOD inline const IMPROVED_ENUM_INHERITED_NAME::EnumType Enum2Inherited(
		const IMPROVED_ENUM_INHERITED_NAME::EnumType& t)
	{ return t; }
	STATIC_METHOD inline const IMPROVED_ENUM_INHERITED_NAME::EnumType Enum2Inherited(
		const EnumType& t)
	{
		switch (t)
		{
			//////////////////////////////////////////
			// With this mini-macro we make ENUMITEM file
			// a CASE list which returns the converted value
			// from one namespace to the other:
#define  ENUMITEM(EnumItem) \
		case EnumItem : return IMPROVED_ENUM_INHERITED_NAME::EnumItem;
#define  ENUMITEM_VALUE(EnumItem, Value) ENUMITEM(EnumItem)
#ifdef   IMPROVED_ENUM_INHERITED_FILE
#include IMPROVED_ENUM_INHERITED_FILE
#endif// IMPROVED_ENUM_INHERITED_FILE
#undef   ENUMITEM_VALUE
#undef   ENUMITEM
			//////////////////////////////////////////
		}
		return IMPROVED_ENUM_INHERITED_NAME::NotValidEnumItem;
	}

	//! Enum iteration to next extended (not inherited):
	//////////////////////////////////////////
	STATIC_METHOD inline const EnumType NextExtendedEnumItem(
		const EnumType& t)
	{
		switch (t)
		{
		case NotValidEnumItem : 
			//////////////////////////////////////////
			// With this mini-macro we make ENUMITEM file/s
			// a CASE list which returns the next item:
#define  ENUMITEM(EnumItem) return EnumItem; case EnumItem : 
#define  ENUMITEM_VALUE(EnumItem, Value) ENUMITEM(EnumItem)
#ifdef   IMPROVED_ENUM_FILE
#include IMPROVED_ENUM_FILE
#else // IMPROVED_ENUM_LIST
			IMPROVED_ENUM_LIST
#endif// IMPROVED_ENUM_FILE
#undef   ENUMITEM_VALUE
#undef   ENUMITEM
				//////////////////////////////////////////
				return NotValidEnumItem;
		}
		return NotValidEnumItem;
	}

	//! Enum iteration to previous extended (not inherited):
	//////////////////////////////////////////
	STATIC_METHOD inline const EnumType PreviousExtendedEnumItem(
		const EnumType& t)
	{
		EnumType tprev = NotValidEnumItem;
		//////////////////////////////////////////
		// With this mini-macro we make ENUMITEM file/s
		// an IF list which returns the previous item:
#define  ENUMITEM(EnumItem) \
	if (t == EnumItem) return tprev; else tprev = EnumItem;
#define  ENUMITEM_VALUE(EnumItem, Value) ENUMITEM(EnumItem)
#ifdef   IMPROVED_ENUM_FILE
#include IMPROVED_ENUM_FILE
#else // IMPROVED_ENUM_LIST
		IMPROVED_ENUM_LIST
#endif// IMPROVED_ENUM_FILE
#undef   ENUMITEM_VALUE
#undef   ENUMITEM
			//////////////////////////////////////////
			return tprev;
	}

	//! The first and the last extended Enums:
	//////////////////////////////////////////
	STATIC_METHOD inline const EnumType FirstExtendedEnumItem()
	{ return NextExtendedEnumItem(NotValidEnumItem); }
	STATIC_METHOD inline const EnumType LastExtendedEnumItem()
	{ return PreviousExtendedEnumItem(NotValidEnumItem); }

	//! Number of extended enum items:
	//////////////////////////////////////////
	STATIC_METHOD inline const int NumberOfExtendedValidEnumItem()
	{
		return 0
			//////////////////////////////////////////
			// With this mini-macro we make ENUMITEM file
			// a counter list:
#define  ENUMITEM(EnumItem) +1
#define  ENUMITEM_VALUE(EnumItem, Value) ENUMITEM(EnumItem)
#ifdef   IMPROVED_ENUM_FILE
#include IMPROVED_ENUM_FILE
#else // IMPROVED_ENUM_LIST
			IMPROVED_ENUM_LIST
#endif// IMPROVED_ENUM_FILE
#undef   ENUMITEM_VALUE
#undef   ENUMITEM
			//////////////////////////////////////////
			;
	}

	//! Enum iteration to next inherited:
	//////////////////////////////////////////
	STATIC_METHOD inline const EnumType NextInheritedEnumItem(
		const EnumType& t)
	{
		switch (t)
		{
		case NotValidEnumItem : 
			//////////////////////////////////////////
			// With this mini-macro we make ENUMITEM file/s
			// a CASE list which returns the next item:
#define  ENUMITEM(EnumItem) return EnumItem; case EnumItem : 
#define  ENUMITEM_VALUE(EnumItem, Value) ENUMITEM(EnumItem)
#include IMPROVED_ENUM_INHERITED_FILE
#undef   ENUMITEM_VALUE
#undef   ENUMITEM
			//////////////////////////////////////////
			return NotValidEnumItem;
		}
		return NotValidEnumItem;
	}

	//! Enum iteration to previous inherited:
	//////////////////////////////////////////
	STATIC_METHOD inline const EnumType PreviousInheritedEnumItem(
		const EnumType& t)
	{
		EnumType tprev = NotValidEnumItem;
		//////////////////////////////////////////
		// With this mini-macro we make ENUMITEM file/s
		// an IF list which returns the previous item:
#define  ENUMITEM(EnumItem) \
	if (t == EnumItem) return tprev; else tprev = EnumItem;
#define  ENUMITEM_VALUE(EnumItem, Value) ENUMITEM(EnumItem)
#include IMPROVED_ENUM_INHERITED_FILE
#undef   ENUMITEM_VALUE
#undef   ENUMITEM
		//////////////////////////////////////////
		return tprev;
	}

	//! The first and the last inherited Enums:
	//////////////////////////////////////////
	STATIC_METHOD inline const EnumType FirstInheritedEnumItem()
	{ return NextInheritedEnumItem(NotValidEnumItem); }
	STATIC_METHOD inline const EnumType LastInheritedEnumItem()
	{ return PreviousInheritedEnumItem(NotValidEnumItem); }

	//! Number of inherited enum items:
	//////////////////////////////////////////
	STATIC_METHOD inline const int NumberOfInheritedValidEnumItem()
	{
		return 0
			//////////////////////////////////////////
			// With this mini-macro we make ENUMITEM file
			// a counter list:
#define  ENUMITEM(EnumItem) +1
#define  ENUMITEM_VALUE(EnumItem, Value) ENUMITEM(EnumItem)
#include IMPROVED_ENUM_INHERITED_FILE
#undef   ENUMITEM_VALUE
#undef   ENUMITEM
			//////////////////////////////////////////
			;
	}

#endif // IMPROVED_ENUM_INHERITED_NAME

	// Free temporary macros:
	///////////////////////////
#undef STATIC_METHOD
#undef ENUM_SEPARATOR
#undef ENUM_TYPE_NAME
#undef PARENT_ENUM_TYPE_NAME
#undef FULL_ENUM_TYPE_NAME
#undef GET_MACRO_STRING
#undef GET_MACRO_STRING_EXPANDED
}
#if defined(IMPROVED_ENUM_SUBCLASS) || defined(IMPROVED_ENUM_SUBCLASS_PARENT)
;
#endif

// Free this file's parameters:
////////////////////////////////
#undef IMPROVED_ENUM_NAME
#undef IMPROVED_ENUM_FILE
#undef IMPROVED_ENUM_LIST
#undef IMPROVED_ENUM_SUBCLASS
#undef IMPROVED_ENUM_SUBCLASS_PARENT
#undef IMPROVED_ENUM_INHERITED_NAME
#undef IMPROVED_ENUM_INHERITED_FILE
// Do not use directives like: #endif (reason above)