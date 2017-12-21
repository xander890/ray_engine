#include "parameter_parser.h"
#include "xercesc/framework/LocalFileFormatTarget.hpp"

#include <iostream>
#include <fstream>

#include <map>
#include <algorithm>

using namespace std;
#include "xercesc/util/PlatformUtils.hpp"
#include "xercesc/dom/DOM.hpp"
#include <xercesc/parsers/XercesDOMParser.hpp>

XERCES_CPP_NAMESPACE_USE


static DOMDocument * document = nullptr;
static XercesDOMParser * configParser = nullptr;

string ConfigParameters::document_file = string("");
std::map<std::string, std::map<std::string, ConfigParameters::NodeElem>>  ConfigParameters::parameters;
std::map<std::string, std::map<std::string, ConfigParameters::NodeElem>>  ConfigParameters::used_parameters;

ConfigParameters::ConfigParameters(void)
{
}


ConfigParameters::~ConfigParameters(void)
{
	free();
}

DOMDocument * createEmptyParameterSheet()
{	
    // Pointer to our DOMImplementation.
    DOMImplementation*    p_DOMImplementation = nullptr;

    // Get the DOM Implementation (used for creating DOMDocuments).
    // Also see: http://www.w3.org/TR/2000/REC-DOM-Level-2-Core-20001113/core.html
    p_DOMImplementation = DOMImplementationRegistry::getDOMImplementation(
             XMLString::transcode("core"));
    // Pointer to our DOMDocument.
    DOMDocument*        pDOMDocument = nullptr;
	pDOMDocument = p_DOMImplementation->createDocument(XMLString::transcode("schemas.example.com/2008/"), 
		XMLString::transcode("ex:Optix_Framework_Parameters"), 0);
	DOMElement * pRootElement = nullptr;
	pRootElement = pDOMDocument->getDocumentElement();

	return pDOMDocument;	
}

DOMDocument* read_from_file(const char * FullFilePath)
{
	
	configParser->setValidationScheme(AbstractDOMParser::Val_Never);
	configParser->setDoNamespaces(false);
	configParser->setDoSchema(false);
	configParser->setLoadExternalDTD(false);
	DOMDocument * doc = nullptr;
	try
	{
		configParser->parse(FullFilePath);
		doc = configParser->getDocument();
	}
	catch( xercesc::XMLException& e )
	{
			char* message = XMLString::transcode( e.getMessage() );
			Logger::error << "Error parsing file: " << std::string(message) << std::endl;
			XMLString::release( &message );
	}
	if (!doc)
	{
		Logger::error << "Unable to parse XML " << FullFilePath << std::endl;
		doc = createEmptyParameterSheet();
	}

	return doc;
}

void ConfigParameters::parse_doc()
{
	DOMElement* elementRoot = document->getDocumentElement();
	if (elementRoot == nullptr || !elementRoot->hasChildNodes()) return;
	DOMNodeList*      children = elementRoot->getChildNodes();
	const  XMLSize_t nodeCount = children->getLength();

	for (XMLSize_t xx = 0; xx < nodeCount; ++xx)
	{
		DOMNode* currentNode = children->item(xx);
		if (currentNode->getNodeType() &&  // true is not nullptr
			currentNode->getNodeType() == DOMNode::ELEMENT_NODE) // is element
		{
			// Found node which is an Element. Re-cast node as element
			DOMElement* currentElement = dynamic_cast< xercesc::DOMElement* >(currentNode);
			const XMLCh * key = currentElement->getTagName();
			auto group = std::string(XMLString::transcode(key));
			DOMNodeList* children2 = currentElement->getChildNodes();
			const XMLSize_t nodes = children2->getLength();
			for (XMLSize_t yy = 0; yy < nodes; ++yy)
			{
				DOMNode* nephew = children2->item(yy);
				if (nephew->getNodeType() &&  // true is not nullptr
					nephew->getNodeType() == DOMNode::ELEMENT_NODE) // is element
				{
					DOMElement* currentElementNephew = dynamic_cast<xercesc::DOMElement*>(nephew);
					const char * v = XMLString::transcode(currentElementNephew->getAttribute(XMLString::transcode("value")));
					const char * k = XMLString::transcode(currentElementNephew->getAttribute(XMLString::transcode("key")));
					const char * c = XMLString::transcode(currentElementNephew->getAttribute(XMLString::transcode("comment")));
					parameters[group][k] = { std::string(v), std::string(c) };
				}
			}
		}
	}
}

DOMNode* find_node(const char * name, DOMElement* elementRoot)
{
	if (elementRoot == nullptr)
		elementRoot = document->getDocumentElement();
	if(elementRoot == nullptr || !elementRoot->hasChildNodes()) return nullptr;
	DOMNodeList*      children = elementRoot->getChildNodes();
	const  XMLSize_t nodeCount = children->getLength();
	XMLCh * n = XMLString::transcode(name);
	static XMLCh * prop = XMLString::transcode("key");

	for( XMLSize_t xx = 0; xx < nodeCount; ++xx )
	{
		DOMNode* currentNode = children->item(xx);
		if( currentNode->getNodeType() &&  // true is not nullptr
			currentNode->getNodeType() == DOMNode::ELEMENT_NODE ) // is element
		{
			// Found node which is an Element. Re-cast node as element
			DOMElement* currentElement = dynamic_cast< xercesc::DOMElement* >( currentNode );


			const XMLCh * key = currentElement->getAttribute(prop);

			if( XMLString::equals(key, n))
			{
				return currentElement;
			}
		}
	}
	return nullptr;
}

DOMNode* find_tag(const char* tag)
{
	DOMElement* elementRoot = document->getDocumentElement();
	if (elementRoot == nullptr || !elementRoot->hasChildNodes()) return nullptr;
	DOMNodeList*      children = elementRoot->getChildNodes();
	const  XMLSize_t nodeCount = children->getLength();
	XMLCh * n = XMLString::transcode(tag);

	for (XMLSize_t xx = 0; xx < nodeCount; ++xx)
	{
		DOMNode* currentNode = children->item(xx);
		if (currentNode->getNodeType() &&  // true is not nullptr
			currentNode->getNodeType() == DOMNode::ELEMENT_NODE) // is element
		{
			// Found node which is an Element. Re-cast node as element
			DOMElement* currentElement = dynamic_cast< xercesc::DOMElement* >(currentNode);
			const XMLCh * key = currentElement->getTagName();

			if (XMLString::equals(key, n))
			{
				return currentElement;
			}
		}
	}
	return nullptr;
}

void ConfigParameters::init(const std::string & document_f)
{
	XMLPlatformUtils::Initialize();
	configParser = new XercesDOMParser;
	document = read_from_file(document_f.c_str());
	parse_doc();
	document_file = string(document_f);
}


void ConfigParameters::free()
{
	document->release();
	XMLPlatformUtils::Terminate();
}

std::string tag(const std::string & s, bool is_end)
{
	return std::string("<") + (is_end ? "/" : "") + s + ">";
}

void ConfigParameters::dump_used_parameters(const std::string& document_f)
{
	stringstream ss;
	ss << "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\" ?>" << std::endl;
	ss << "<ex:Optix_Framework_Parameters xmlns:ex=\"schemas.example.com/2008/\">" << std::endl;
	for (const auto & pair : used_parameters)
	{
		ss << "    " << tag(pair.first, false) << std::endl;
		for(const auto & pair2 : pair.second)
		{
			ss << "        <property " << "key=\"" << pair2.first << "\" " << "value=\"" << pair2.second.value << "\" " << "comment=\"" << pair2.second.comment << "\"" << "/>" << std::endl;
		}
		ss << "    " << tag(pair.first, true) << std::endl;
	}
	ss << "</ex:Optix_Framework_Parameters>" << std::endl;
	std::string res = ss.str();
	std::ofstream ofs_data(document_f);
	if (ofs_data.bad())
	{
		Logger::error << "Unable to open file " << document_f << endl;
		return;
	}
	ofs_data << res << std::endl;
	ofs_data.close();
}


void ConfigParameters::override_parameters(std::vector<std::string>& override_argv)
{
	for (int i = 0; i < override_argv.size(); i++)
	{
		std::string a = override_argv[i];
		const size_t sep = a.find_first_of("/");
		if (sep != std::string::npos)
		{
			const std::string tag = a.substr(0, sep);
			const std::string elem = a.substr(sep + 1, a.size());
			i++;
			auto a2 = override_argv[i];
			a2.erase(std::remove(a2.begin(), a2.end(), '\"'), a2.end());
			if (parameters.count(tag) != 0 && parameters[tag].count(elem) != 0)
			{
				std::string old = parameters[tag][elem].value;
				parameters[tag][elem].value = a2;
				Logger::debug << "Overriding parameter " << tag << "/" << elem << " from " << old << " to " << a2 << std::endl;
			}
			else
			{
				Logger::warning << "Trying to override non existing parameter: " << elem << " (tag " << tag << ")" << a2 << std::endl;
				parameters[tag][elem] = { a2 , ""};
			}
		}
	}
}
