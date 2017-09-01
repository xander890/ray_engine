#ifndef parameter_parser_h__
#define parameter_parser_h__

#include "parserstringhelpers.h"
#include "xercesc/util/PlatformUtils.hpp"
#include "xercesc/dom/DOM.hpp"
#include <xercesc/parsers/XercesDOMParser.hpp>

#include "logger.h"

using namespace std;
XERCES_CPP_NAMESPACE_USE

class ParameterParser
{
public:	
	static void init(const std::string & document_f);
	static void free();

	static void set_document(const char * newPath);;
	static void write_to_file();
	static void write_to_file(const char * FullFilePath);

	//template <typename T> 
	//static T get_parameter(const char * n, T default_value, const char * comment = "")
	//{

	//	const XMLCh * name = XMLString::transcode(n);
	//	DOMNode * node = find_node(n);
	//	if(node != nullptr)
	//	{
	//		DOMElement* currentElement = dynamic_cast< xercesc::DOMElement* >( node );
	//		const char * v = XMLString::transcode(currentElement->getAttribute( XMLString::transcode("value")));
	//		return tovalue<T>(string(v));
	//	}
	//	DOMElement* elementRoot = document->getDocumentElement();
	//	const XMLCh * val = XMLString::transcode(tostring<T>(default_value).c_str());

	//	DOMElement * child = document->createElement(XMLString::transcode("property"));
	//	child->setAttribute(XMLString::transcode("key"),name);
	//	child->setAttribute(XMLString::transcode("value"),val);
	//	if(strcmp(comment,"") != 0)
	//		child->setAttribute(XMLString::transcode("comment"),XMLString::transcode(comment));
	//	if (elementRoot)
	//		elementRoot->appendChild(child);
	//	return default_value;
	//}

	template <typename T>
	static T get_parameter(const char * tag, const char * n, T default_value, const char * comment = "")
	{
		const XMLCh * name = XMLString::transcode(n);
		const XMLCh * tags = XMLString::transcode(tag);
		DOMElement* elementRoot = document->getDocumentElement();

		DOMNode * tag_node = find_tag(tag);
		if (tag_node == nullptr)
		{
			//Logger::error <<  "Tag Node ", tag, " not found." });
			DOMElement * child_tag = document->createElement(tags);
			elementRoot->appendChild(child_tag);
		}
		DOMElement * tag_element = dynamic_cast< xercesc::DOMElement* >(tag_node);

		DOMNode * node = find_node(n, tag_element);

		if (node != nullptr)
		{
			DOMElement* currentElement = dynamic_cast< xercesc::DOMElement* >(node);
			const char * v = XMLString::transcode(currentElement->getAttribute(XMLString::transcode("value")));
			//Logger::error <<  "Node ", n, " found. Value ", v });
			return tovalue<T>(string(v));
		}
		const XMLCh * val = XMLString::transcode(tostring<T>(default_value).c_str());

		DOMElement * child = document->createElement(XMLString::transcode("property"));
		child->setAttribute(XMLString::transcode("key"), name);
		child->setAttribute(XMLString::transcode("value"), val);
		if (strcmp(comment, "") != 0)
			child->setAttribute(XMLString::transcode("comment"), XMLString::transcode(comment));



		if (tag_element)
			tag_element->appendChild(child);
		return default_value;
	}

	/*template <typename T>
	static void set_parameter(const char * name, T value, const char * comment = "")
	{
		DOMNode * node = find_node(name);
		DOMElement* child;
		if(node)
		{
			child = dynamic_cast< xercesc::DOMElement* >( node );
		}
		else
		{
			DOMElement* elementRoot = document->getDocumentElement();
			child = document->createElement(XMLString::transcode("property"));
			if (elementRoot)
				elementRoot->appendChild(child);
		}

		child->setAttribute(XMLString::transcode("key"),XMLString::transcode(name));
		child->setAttribute(XMLString::transcode("value"),XMLString::transcode(tostring<T>(value).c_str()));
		if(strcmp(comment,"") != 0)
			child->setAttribute(XMLString::transcode("comment"),XMLString::transcode(comment));

	}*/



protected:
	ParameterParser(void);
	~ParameterParser(void);

	static DOMNode* find_node(const char * name, DOMElement* root_element = nullptr);
	static DOMNode* find_tag(const char * tag);
	static DOMDocument * createEmptyParameterSheet();
	static DOMDocument * read_from_file(const char * FullFilePath);
	static DOMDocument * document;
	static XercesDOMParser * configParser;
	static std::string document_file;


	static std::string config_folder;
	
};
#endif // parameter_parser_h__
