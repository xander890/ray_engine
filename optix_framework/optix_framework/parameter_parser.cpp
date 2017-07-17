#include "parameter_parser.h"
#include "xercesc/framework/LocalFileFormatTarget.hpp"

#include <iostream>

XercesDOMParser * ParameterParser::configParser = nullptr;
DOMDocument * ParameterParser::document = nullptr;
string ParameterParser::document_file = string("");
string ParameterParser::config_folder = string("");

ParameterParser::ParameterParser(void)
{
}


ParameterParser::~ParameterParser(void)
{
	free();
}

DOMDocument * ParameterParser::createEmptyParameterSheet()
{	
    // Pointer to our DOMImplementation.
    DOMImplementation*    p_DOMImplementation = nullptr;

    // Get the DOM Implementation (used for creating DOMDocuments).
    // Also see: http://www.w3.org/TR/2000/REC-DOM-Level-2-Core-20001113/core.html
    p_DOMImplementation = DOMImplementationRegistry::getDOMImplementation(
             XMLString::transcode("core"));
    // Pointer to our DOMDocument.
    DOMDocument*        pDOMDocument = nullptr;
	pDOMDocument = p_DOMImplementation->createDocument(L"schemas.example.com/2008/", 
		L"ex:Optix_Framework_Parameters", 0);
	DOMElement * pRootElement = nullptr;
	pRootElement = pDOMDocument->getDocumentElement();

	return pDOMDocument;	
}

void ParameterParser::write_to_file(const char * FullFilePath )
{
	DOMImplementation	*pImplement	= nullptr;
	DOMLSSerializer *pSerializer	= nullptr; // @DOMWriter
	LocalFileFormatTarget *pTarget	= nullptr; 


	//Return the first registered implementation that has the desired features. In this case, we are after
	//a DOM implementation that has the LS feature... or Load/Save.

	pImplement = DOMImplementationRegistry::getDOMImplementation(L"LS");

	//From the DOMImplementation, create a DOMWriter.
	//DOMWriters are used to serialize a DOM tree [back] into an XML document.

	pSerializer = ((DOMImplementationLS*)pImplement)->createLSSerializer(); //@createDOMWriter();

	//This line is optional. It just sets a feature of the Serializer to make the output
	//more human-readable by inserting line-feeds, without actually inserting any new elements/nodes
	//into the DOM tree. (There are many different features to set.) Comment it out and see the difference.

	// @pSerializer->setFeature(XMLUni::fgDOMWRTFormatPrettyPrint, true); // 
	DOMLSOutput *pOutput = ((DOMImplementationLS*)pImplement)->createLSOutput();
	DOMConfiguration *pConfiguration = pSerializer->getDomConfig();

	// Have a nice output
	if (pConfiguration->canSetParameter(XMLUni::fgDOMWRTFormatPrettyPrint, true))
		pConfiguration->setParameter(XMLUni::fgDOMWRTFormatPrettyPrint, true); 

	try
	{
		pTarget = new LocalFileFormatTarget(FullFilePath);
	}
	catch (xercesc::XMLException& e)
	{
		char* message = XMLString::transcode(e.getMessage());
		std::cerr << "Error parsing file: " << message << std::endl;
		XMLString::release(&message);
		return;
	}
	pOutput->setByteStream(pTarget);

	
	// @pSerializer->write(pDOMDocument->getDocumentElement(), pOutput); // missing header "<xml ...>" if used
	
	try
	{
		pSerializer->write(document, pOutput); 
	}
	catch( xercesc::XMLException& e )
	{
		char* message = XMLString::transcode( e.getMessage() );
		std::cerr << "Error parsing file: " << message << std::endl;
		XMLString::release( &message );
	}

	delete pTarget;
	pOutput->release();
	pSerializer->release();
}

void ParameterParser::write_to_file()
{
	write_to_file((config_folder + document_file).c_str());
}

DOMDocument* ParameterParser::read_from_file(const char * FullFilePath)
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
			std::cerr << "Error parsing file: " << message << std::endl;
			XMLString::release( &message );
	}
	if(!doc)
		doc = createEmptyParameterSheet();
	return doc;
}

DOMNode* ParameterParser::find_node(const char * name, DOMElement* elementRoot)
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

DOMNode* ParameterParser::find_tag(const char* tag)
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

void ParameterParser::init(const std::string & document_f)
{
	XMLPlatformUtils::Initialize();
	configParser = new XercesDOMParser;
	document = read_from_file(document_f.c_str());
	document_file = string(document_f);
	config_folder = get_parameter<std::string>("folders","config_folder", std::string("./"), "The folder where to look for all the configuration files.");
}


void ParameterParser::free()
{
	write_to_file(document_file.c_str());
	document->release();
	XMLPlatformUtils::Terminate();
}

void ParameterParser::set_document(const char * newPath)
{
	write_to_file((config_folder + document_file).c_str());
	document_file = string(newPath);
	if(document)
		document->release();
	document = read_from_file((config_folder + document_file).c_str());
}

